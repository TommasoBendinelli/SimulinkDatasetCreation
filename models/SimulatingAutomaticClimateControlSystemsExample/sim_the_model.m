function res = sim_the_model(args)
% Utility function to simulate a Simulink model (named 'the_model') with
% the specified parameter and input signal values.
% 
% Inputs:
%    StopTime:
%       Simulation stop time, default is nan
%    TunableParameters
%       A struct where the fields are the tunanle referenced
%       workspace variables with the values to use for the
%       simulation.
%    ExternalInput:
%       External Input signal, defualt is empty 
%    Values of nan or empty for the above inputs indicate that sim should
%    run with the default values set in the model.
% 
% Outputs:
%    res: A structure with the time and data values of the logged signals.

% By: Tommaso Bendinelli 11-Sep-2025, heavily based on code from Murali Yeddanapudi, 20-Feb-2022

arguments
    args.uST (1,1) double = 0.1
    args.StopTime (1,1) double = nan
    args.TunableParameters = []
    args.ExternalInput = []   % map key/value
    args.ConfigureForDeployment (1,1) {mustBeNumericOrLogical} = true
    args.OutputFcn (1,1)  {mustBeFunctionHandle} = @emptyFunction
    args.OutputFcnDecimation (1,1) {mustBeInteger, mustBePositive} = 1
    args.DiagramDataPath (1,1) string = "."   % <-- NEW: path argument
end
    import mlreportgen.report.*
    import slreportgen.report.*
    import slreportgen.finder.*

    % --- Load the model (don’t assume it’s open when using MATLAB Engine) -----
    model_name = 'simulink_model';              % base name (no .slx)
    mdlfile = which([model_name '.slx']);
    if isempty(mdlfile)
        error("sim_the_model:ModelNotFound", ...
            "Could not find %s.slx on the MATLAB path. Current folder: %s", ...
            model_name, pwd);
    end

    if ~bdIsLoaded(model_name)
        load_system(model_name);
    end
    si = Simulink.SimulationInput(model_name);
    si = si.setVariable('uST', args.uST);
    % Find all subsystems in the model (excluding library-linked ones)
    set(gcf,'PaperType','A4')
    subs = find_system(model_name, ...
    'LookUnderMasks','all', ...
    'FollowLinks','off', ...
    'BlockType','SubSystem');
    subs = [model_name; subs];
    % Optional: set page layout for the model prints
    set_param(model_name, 'PaperOrientation','landscape');  % 'portrait' or 'landscape'
    set_param(model_name, 'PaperPositionMode','auto');
    
    for k = 1:numel(subs)
        s = subs{k};

        % Skip library-linked subsystems if any slipped through
        % if ~strcmp(get_param(s,'LinkStatus'),'none'); continue; end

        % Make a safe filename that reflects the full path
        safeName = regexprep(s, '[^a-zA-Z0-9_-]', '_');  % replace / spaces etc.
        pngPath  = fullfile(args.DiagramDataPath, [safeName '.png']);

        % Print the subsystem diagram to PNG
        % Note: pass the system path via -s<path>
        try
            print(['-s' s], '-dpng', pngPath);
            fprintf('Saved: %s\n', pngPath);
        catch ME
            warning('Could not print %s: %s', s, ME.message);
        end
    end
    
    report_generator_Path = fullfile(args.DiagramDataPath, "_SystemIO_Report.pdf");
    rpt = slreportgen.report.Report(report_generator_Path);
    
    %% Load the StopTime into the SimulationInput object
    if ~isnan(args.StopTime)
        si = si.setModelParameter('StopTime', num2str(args.StopTime));
    end
    
    %% Load the specified tunable parameters into the simulation input object
    if isstruct(args.TunableParameters) 
        tpNames = fieldnames(args.TunableParameters);
        for itp = 1:numel(tpNames)
            tpn = tpNames{itp};
            tpv = args.TunableParameters.(tpn);
            si = si.setVariable(tpn, tpv);
        end
    end
    
    %%ExternalInput = [0.0  2.0  0.0  0.0  0.0 -2.0 -2.0  0.0;
    %%             1.0  0.0  1.0  1.0  1.0  0.0  0.0  1.0];
    
    % Example with 50,001 samples
    % u1 = 23 * ones(1, 50001);   % row vector of all 23
    % u2 = 28 * ones(1, 50001);   % row vector of all 28
    % ExternalInput = {
    %        'Input Set Temperature', u1;
    %        'Input External Temperature', u2;
    %        'Error Efficiency', u1;
    %    };

   
    %% disp(args.ExternalInput)
    ExternalInput = args.ExternalInput;
    disp(ExternalInput);
    %% Load the external input into the SimulationInput object
    if ~isempty(ExternalInput)
        % In the model, the external input u is a discrete signal with sample
        % time 'uST'. Hence the time points where it is sampled are set, i.e.,
        % they are multiples of uST: 0, uST, 2*uST, 3*uST, .. We only specify
        % the data values here using the struct with empty time field as
        % described in Guy's blog post:
        % https://blogs.mathworks.com/simulink/2012/02/09/using-discrete-data-as-an-input-to-your-simulink-model/
        
        % Find all Inport blocks
        inports = find_system(model_name, 'BlockType', 'Inport');
        top_inports = inports( cellfun(@(x) count(x,'/')==1, inports) );
        names = cellfun(@(x) extractAfter(x, '/'), top_inports, 'UniformOutput', false);
        
        % Check that the names matchex the names of ExternalInput
        extData = cell(numel(names),1);
        
        for i = 1:numel(names)
            key = names{i};      
            disp(key); % the i-th Inport name (model order)
            disp(strcmp(ExternalInput(:,1), key));
            idx = find(strcmp(ExternalInput(:,1), key), 1);
            if numel(idx)
                error('Missing ExternalInput for Inport: %s', key);
            end
            vals = ExternalInput.(key);
            % vals = ExternalInput{idx,2}(:);          % column vector
            extData{i} = vals; %%%
            % struct( ...
                % 'time', [], ...
                % 'signals', struct('values', vals, 'dimensions', 1) );
        end
        
        % Extract the signals (second column of ExternalInput)
        signalMatrix = cell2mat(extData);   % gives [50001 x 3] because cell2mat stacks column-wise
        N = size(signalMatrix, 2);   % number of samples
        nSignals = size(signalMatrix, 1);  % number of Inports/signals
     
        uStruct.time = [];
        uStruct.signals.dimensions = 2;
        % values needs to be column vector
        for k = 1:nSignals
            uStruct.signals(k).values     = signalMatrix(k, :).';  % N×1
            uStruct.signals(k).dimensions = 1;                      % scalar input
        end

        si.ExternalInput = uStruct;
    end


    %% OutputFcn
    prevSimTime = nan;
    function locPostStepFcn(simTime)
        so = simulink.compiler.getSimulationOutput('the_model');
        res = extractResults(so, prevSimTime);
        stopRequested = feval(args.OutputFcn, simTime, res);
        if stopRequested
            simulink.compiler.stopSimulation('the_model');
        end
        prevSimTime = simTime;
    end
    if ~isequal(args.OutputFcn, @emptyFunction)
        si = simulink.compiler.setPostStepFcn(si, @locPostStepFcn, ...
            'Decimation', args.OutputFcnDecimation);
    end
    
    %% call sim
    so = sim(si);
    
    %% Extract the simulation results
    % Package the time and data values of the logged signals into a structure
    res = extractResults(so,nan);

end % sim_the_model_using_matlab_runtime

function res = extractResults(so, prevSimTime)
    % Always return a struct, even if empty
    res = struct();

    % 1) Get the signal-logging Dataset regardless of its property name
    ds = getSignalLoggingDataset(so);  % helper below

    % 2) Iterate dataset elements (each is a Simulink.SimulationData.Signal)
    n = ds.numElements;
    for i = 1:n
        sig = ds.getElement(i);
        ts  = sig.Values;     % timeseries
        if isempty(ts) || ~isa(ts,'timeseries'); continue; end

        % Use the signal's logical name; sanitize for struct field name
        name = sig.Name;
        if isempty(name); name = sprintf('Signal_%d', i); end
        fname = matlab.lang.makeValidName(name);

        % 3) Slice by prevSimTime if provided
        if isfinite(prevSimTime)
            idx = ts.Time > prevSimTime;
        else
            idx = true(size(ts.Time));
        end

        % 4) Store into result struct
        res.(fname).Time = ts.Time(idx);
        % Data can be vector/matrix; keep shape
        res.(fname).Data = ts.Data(idx, :);
    end
end

function ds = getSignalLoggingDataset(so)
    % Try the configured Signal Logging name first (most precise)
    ds = [];
    try
        mdl = so.SimulationMetadata.ModelInfo.ModelName;
        logName = get_param(mdl, 'SignalLoggingName');  % e.g. 'logsout' or custom
        if isprop(so, logName)
            cand = so.(logName);
            if isa(cand, 'Simulink.SimulationData.Dataset')
                ds = cand;
                return
            end
        end
    catch
        % fall through to scan
    end

    % Fallback: scan SimulationOutput properties for the first Dataset
    props = who(so);  % editable/logged props
    for k = 1:numel(props)
        val = so.(props{k});
        if isa(val, 'Simulink.SimulationData.Dataset')
            % Prefer a Dataset that looks like signal logging (optional heuristic)
            % If you want strictly the *first* dataset, just return here.
            ds = val;
            return
        end
    end

    error('extractResults:NoDatasetFound', ...
          'No Simulink.SimulationData.Dataset found in the SimulationOutput.');
end

function mustBeFunctionHandle(fh)
    if ~isa(fh,'function_handle') && ~ischar(fh) && ~isstring(fh)
        throwAsCaller(error("Must be a function handle"));
    end
end




function emptyFunction
end
