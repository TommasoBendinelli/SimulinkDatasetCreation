function res = sim_the_model(args)
% Utility function to simulate a Simulink model with piecewise runs:
% whenever a TimeVaryingParameter is scheduled to change, we stop,
% apply the change (between runs), and resume from the saved operating point.
%
% Inputs:
%   uST, StopTime, TunableParameters, ExternalInput, TimeVaryingParameters,
%   ConfigureForDeployment, DiagramDataPath, Debug
%
% Outputs:
%   res: struct with concatenated logged signals across all segments
%        and the final OperatingPoint in res.OperatingPoint.
%
% By: Tommaso Bendinelli 11-Sep-2025 (segmented-run version)

arguments
    args.uST (1,1) double = 0.1
    args.StopTime (1,1) double = nan
    args.TunableParameters = []
    args.ExternalInput = []           % struct of inportName -> [N x d] values
    args.TimeVaryingParameters = []   % struct with fields: identifier,key,time,values,seen (cell arrays)
    args.ConfigureForDeployment (1,1) {mustBeNumericOrLogical} = true
    args.DiagramDataPath (1,1) string = "."
    args.Debug (1,1) logical = true
end

    assignin('base','uST',args.uST);

    % --- Model discovery / load ---
    model_name = 'simulink_model';
    mdlfile = which([model_name '.slx']);
    if isempty(mdlfile)
        error("sim_the_model:ModelNotFound", ...
              "Could not find %s.slx on the MATLAB path. Current folder: %s", ...
              model_name, pwd);
    end
    if ~bdIsLoaded(model_name)
        load_system(model_name);
    end

    % --- Resolve StopTime (absolute) ---
    if ~isnan(args.StopTime)
        Tfinal = args.StopTime;
    else
        st = get_param(model_name,'StopTime');
        Tfinal = str2double(st);
        if isnan(Tfinal)
            error("sim_the_model:StopTimeRequired", ...
                  "Provide a numeric StopTime or set a numeric model StopTime.");
        end
    end
    if Tfinal <= 0
        error("sim_the_model:BadStopTime", "StopTime must be > 0.");
    end

    % --- Create diagram output folder if needed ---
    if ~isfolder(args.DiagramDataPath)
        mkdir(args.DiagramDataPath);
    end

    % --- (Optional) Save time-varying params to file or load a debug default ---
    if ~isempty(args.TimeVaryingParameters)
        tvFile = fullfile(args.DiagramDataPath, "time_varying_params.mat");
        TimeVaryingParameters = args.TimeVaryingParameters;
        S = struct('TimeVaryingParameters', TimeVaryingParameters);
        save(tvFile, '-struct', 'S');
    elseif args.Debug
        tvFile = "/Users/tbe/repos/IndustrialRootAnalysisBench/data/MassSpringDamperWithController/20250925_111304__943f9f21/diagram/time_varying_params.mat";
        if isfile(tvFile)
            L = load(tvFile);
            if isfield(L,'TimeVaryingParameters')
                TimeVaryingParameters = L.TimeVaryingParameters;
            else
                warning("Debug load: 'TimeVaryingParameters' not found in %s", tvFile);
                TimeVaryingParameters = [];
            end
            args.StopTime = 100;
            Tfinal = 100;
        else
            warning("Time-varying parameter file not found: %s", tvFile);
            TimeVaryingParameters = [];
        end
    else
        TimeVaryingParameters = [];
    end

    % --- Snapshot model diagrams (best-effort; non-fatal) ---
    try
        subs = find_system(model_name, 'LookUnderMasks','all', 'FollowLinks','off', 'BlockType','SubSystem');
        subs = [{model_name}; subs(:)];
        set_param(model_name, 'PaperOrientation','landscape', 'PaperPositionMode','auto');
        for k = 1:numel(subs)
            s = subs{k};
            safeName = regexprep(s, '[^a-zA-Z0-9_-]', '_');
            pngPath  = fullfile(args.DiagramDataPath, [safeName '.png']);
            try
                print(['-s' s], '-dpng', pngPath);
                fprintf('Saved: %s\n', pngPath);
            catch ME
                warning('Could not print %s: %s', s, ME.message);
            end
        end
    catch
        % ignore
    end

    % --- Apply initial values for each time-varying parameter (t=0) ---
    if ~isempty(TimeVaryingParameters)
        for k = 1:numel(TimeVaryingParameters.identifier)
            identifier = TimeVaryingParameters.identifier{k};
            key        = TimeVaryingParameters.key{k};
            times      = TimeVaryingParameters.time{k};
            values     = TimeVaryingParameters.values{k};
            if iscell(values), values = values{1}; end

            % pick the value scheduled at t<=0 if present; otherwise first value
            idx0 = find(times <= 0);
            if ~isempty(idx0)
                v0 = values(idx0(end));
            else
                v0 = values(1);
            end
            set_param(identifier, key, num2str(v0));
            % ensure 'seen' is initialized
            if ~isfield(TimeVaryingParameters,'seen') || numel(TimeVaryingParameters.seen) < k || isempty(TimeVaryingParameters.seen{k})
                TimeVaryingParameters.seen{k} = zeros(size(times));
            end
            % mark the last t<=0 event as seen (if any)
            if ~isempty(idx0)
                m = false(size(times));
                m(idx0(end)) = true;
                TimeVaryingParameters.seen{k}(m) = 1;
            end
        end
        set_param(bdroot(model_name),'SimulationCommand','update');
    end

    % --- Prepare External Input (explicit time stamps; safer for segmented runs) ---
    uStruct = [];
    if ~isempty(args.ExternalInput)
        % Get top-level inports
        inports = find_system(model_name, 'SearchDepth',1, 'BlockType','Inport');
        inNames = cellfun(@(x) get_param(x,'Name'), inports, 'UniformOutput', false);

        % Normalize ExternalInput to struct (supports containers.Map)
        ExternalInput = normalizeExternalInput(args.ExternalInput);

        % Validate presence and lengths; build time vector based on uST
        % Expect same number of samples for all provided signals
        N = [];
        for i = 1:numel(inNames)
            nm = inNames{i};
            if ~isfield(ExternalInput, nm)
                error("ExternalInputMissing:Inport", "Missing ExternalInput for Inport: %s", nm);
            end
            vals = ExternalInput.(nm);
            if isvector(vals), vals = vals(:); end
            if isempty(N)
                N = size(vals,1);
            elseif size(vals,1) ~= N
                error("ExternalInputMismatch:Length", "All ExternalInput signals must share the same number of samples.");
            end
        end
        if N < 1
            error("ExternalInputEmpty", "ExternalInput provided but contains no samples.");
        end

        % Ensure inputs cover Tfinal
        tVec = (0:N-1).' * args.uST;
        if tVec(end) < Tfinal
            error("ExternalInputTooShort", ...
                 "ExternalInput length (%.3fs) is shorter than StopTime (%.3fs).", tVec(end), Tfinal);
        end

        % Build uStruct with explicit time
        uStruct.time = tVec;
        for i = 1:numel(inNames)
            nm = inNames{i};
            vals = ExternalInput.(nm);
            vals = ensure2D(vals);  % N x D
            uStruct.signals(i).values     = vals; %#ok<AGROW>
            uStruct.signals(i).dimensions = size(vals,2); %#ok<AGROW>
        end
    end

    % --- Build change schedule (absolute times) ---
    changeTimes = [];
    if ~isempty(TimeVaryingParameters)
        for k = 1:numel(TimeVaryingParameters.identifier)
            times = TimeVaryingParameters.time{k};
            if isempty(times),  continue; end
            % consider only changes after current (0) and up to Tfinal
            changeTimes = [changeTimes; times(:)]; %#ok<AGROW>
        end
        % unique, sorted, keep only (0, Tfinal]
        changeTimes = unique(changeTimes);
        changeTimes = changeTimes(changeTimes > 0 & changeTimes <= Tfinal);
    end

    % --- Segment boundaries: each change triggers a stop->apply->resume ---
    % We'll run segments up to each change time; after each segment we apply
    % the value(s) scheduled at exactly that time; then continue.
    segStops = unique([changeTimes(:); Tfinal]);
    prevStop = 0;

    % --- Prepare accumulation of results ---
    res = struct();
    op = [];   % operating point carried between segments
    
    % Unlock compile-time params, set OP saving once, then enter Fast Restart
    set_param(model_name,'FastRestart','off');
    set_param(model_name,'SaveOperatingPoint','on');   % must stay constant in FR
    % (Optional performance) choose fixed-step solver/logging options here
    set_param(model_name,'FastRestart','on');
        
    % --- Iterate segments ---
    for iseg = 1:numel(segStops)
        disp(iseg)
        tStop = segStops(iseg);

        % Prepare SimulationInput for this segment
        si = Simulink.SimulationInput(model_name);
        si = si.setVariable('uST', args.uST);
        si = si.setModelParameter('StopTime', num2str(tStop));
        % si = si.setModelParameter('SaveOperatingPoint','on');

        % Set tunable parameters
        if isstruct(args.TunableParameters)
            tpNames = fieldnames(args.TunableParameters);
            for itp = 1:numel(tpNames)
                tpn = tpNames{itp};
                tpv = args.TunableParameters.(tpn);
                si = si.setVariable(tpn, tpv);
            end
        end

        % External input (explicit time)
        if ~isempty(uStruct)
            si = si.setExternalInput(uStruct);
        end

        % Resume from prior operating point if available
        if ~isempty(op)
            try
                si = si.setInitialState(op);
            catch ME
                warning("Incompatible operating point (%s). This segment will start cold.", ME.message);
            end
        end

        % Run the segment
        try
            so = sim(si);
        catch ME
            % If initial state caused failure, retry cold once
            if contains(ME.message, 'initial') || contains(ME.message,'OperatingPoint')
                warning("Retrying segment %.6g without initial state: %s", tStop, ME.message);
                siNoOP = removeInitialState(si);
                so = sim(siNoOP);
            else
                rethrow(ME);
            end
        end

        % Accumulate results (slice to avoid duplicate samples at the join)
        segRes = extractResults(so, prevStop);
        res = mergeResults(res, segRes);

        % Stash operating point for next segment
        if isprop(so, 'xFinal') && ~isempty(so.xFinal)
            op = so.xFinal;
        else
            op = [];
        end

        % After hitting tStop, apply any changes scheduled at exactly tStop
        if iseg < numel(segStops)  % no need to apply after the last (unless you want to stage post-final changes)
            [TimeVaryingParameters, didChange] = applyChangesAtTime(TimeVaryingParameters, tStop);
            if didChange
                % Recompile after parameter edits
                set_param(bdroot(model_name),'SimulationCommand','update');
            end
        end

        prevStop = tStop;
    end

    % Expose final OP to caller
    res.OperatingPoint = op;
set_param(model_name,'FastRestart','off');
end


% === Helpers ===============================================================

function ExternalInput = normalizeExternalInput(E)
    if isa(E, 'containers.Map')
        ExternalInput = struct();
        k = E.keys;
        for i = 1:numel(k)
            ExternalInput.(k{i}) = E(k{i});
        end
    elseif isstruct(E)
        ExternalInput = E;
    else
        error("ExternalInputType", "ExternalInput must be a struct or containers.Map.");
    end
end

function M = ensure2D(v)
    % Returns N x D
    if isvector(v)
        M = v(:);
    else
        M = v;
    end
end

function si2 = removeInitialState(si1)
    % Remove initial state from a SimulationInput (best-effort)
    si2 = si1;
    try
        si2 = si2.setInitialState([]); % newer releases accept []
    catch
        % no-op if not supported; construct a fresh one would be plan B
    end
end

function tf = appeq(a,b)
    % approximate equality for event times
    tf = abs(a - b) <= max(1e-9, eps(max(abs(a),abs(b))) * 10);
end

function [TVP, didChange] = applyChangesAtTime(TVP, tEvent)
    didChange = false;
    if isempty(TVP); return; end
    for k = 1:numel(TVP.identifier)
        identifier = TVP.identifier{k};
        key        = TVP.key{k};
        times      = TVP.time{k};
        values     = TVP.values{k};
        if iscell(values), values = values{1}; end
        if ~isfield(TVP,'seen') || isempty(TVP.seen{k})
            TVP.seen{k} = zeros(size(times));
        end

        % indices that are exactly at tEvent and not yet applied
        idx = find(arrayfun(@(tt) appeq(tt, tEvent), times) & TVP.seen{k}==0);
        if isempty(idx), continue; end

        % If multiple at same time, keep the last
        idx = idx(end);
        v = values(idx);

        % Apply change
        set_param(identifier, key, num2str(v));
        TVP.seen{k}(idx) = 1;
        didChange = true;
    end
end

function res = extractResults(so, prevSimTime)
    % Always return a struct, even if empty
    res = struct();

    ds = getSignalLoggingDataset(so);
    n = ds.numElements;
    for i = 1:n
        sig = ds.getElement(i);
        ts  = sig.Values;
        if isempty(ts) || ~isa(ts,'timeseries'); continue; end

        name = sig.Name;
        if isempty(name); name = sprintf('Signal_%d', i); end
        fname = matlab.lang.makeValidName(name);

        if isfinite(prevSimTime)
            idx = ts.Time > prevSimTime;
        else
            idx = true(size(ts.Time));
        end

        res.(fname).Time = ts.Time(idx);
        res.(fname).Data = ts.Data(idx, :);
    end
end

function res = mergeResults(res, segRes)
    if isempty(fieldnames(segRes))
        return
    end
    if isempty(fieldnames(res))
        res = segRes;
        return
    end
    f = fieldnames(segRes);
    for i = 1:numel(f)
        fn = f{i};
        if ~isfield(res, fn)
            res.(fn) = segRes.(fn);
        else
            res.(fn).Time = [res.(fn).Time; segRes.(fn).Time];
            res.(fn).Data = [res.(fn).Data; segRes.(fn).Data];
        end
    end
end

function ds = getSignalLoggingDataset(so)
    % Try configured Signal Logging name first
    ds = [];
    try
        mdl = so.SimulationMetadata.ModelInfo.ModelName;
        logName = get_param(mdl, 'SignalLoggingName');  % e.g. 'logsout'
        if isprop(so, logName)
            cand = so.(logName);
            if isa(cand, 'Simulink.SimulationData.Dataset')
                ds = cand;  return
            end
        end
    catch
        % fall through
    end

    % Fallback: first Dataset on SimulationOutput
    props = who(so);
    for k = 1:numel(props)
        val = so.(props{k});
        if isa(val, 'Simulink.SimulationData.Dataset')
            ds = val;  return
        end
    end
    error('extractResults:NoDatasetFound', ...
        'No Simulink.SimulationData.Dataset found in the SimulationOutput.');
end