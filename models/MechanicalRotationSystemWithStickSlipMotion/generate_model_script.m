function scriptText = exportSimulinkToScript(inModel, outModel)
% Recursively export a Simulink model to a MATLAB script that rebuilds it.
% Safe for filenames w/ .slx and diagram names; no string '+' used.

    if nargin < 1 || isempty(inModel)
        error('Provide a model name or .slx file path.');
    end
    inModel = char(inModel);
    [p, base, ext] = fileparts(inModel);
    if isempty(ext)
        sysName    = base;
        fileToLoad = inModel;
    else
        sysName    = base;                 % diagram name has no extension
        fileToLoad = fullfile(p, [base ext]);
    end
    if ~bdIsLoaded(sysName)
        try
            load_system(fileToLoad);
        catch
            load_system(sysName);
        end
    end

    if nargin < 2 || isempty(outModel)
        outModel = matlab.lang.makeValidName([sysName '_rebuild']);
    else
        outModel = char(matlab.lang.makeValidName(string(outModel)));
    end

    buf = '';
    emit = @(s) [s sprintf('\n')];

    % Create empty model
    buf = [buf emit(sprintf('new_system(''%s'');', outModel))];
    buf = [buf emit(sprintf('open_system(''%s'');', outModel))];

    % Recurse from top level
    buf = [buf emit(emitSystem(sysName, outModel))];

    % Arrange & save
    buf = [buf emit(sprintf('Simulink.BlockDiagram.arrangeSystem(''%s'');', outModel))];
    buf = [buf emit(sprintf('save_system(''%s'');', outModel))];

    scriptText = buf;

    % -------------- helpers ----------------
    function out = emitSystem(srcSystem, tgtSystem)
        out = '';

        % --- blocks (this level) ---
        blks = find_system(srcSystem,'SearchDepth',1,'Type','Block');
        for k = 1:numel(blks)
            b = blks{k};
            [~, name] = fileparts(b);
            tgtPath   = [tgtSystem '/' name];

            % library or built-in source
            srcRef = get_param(b,'ReferenceBlock');
            if isempty(srcRef)
                bt = get_param(b,'BlockType');
                if strcmpi(bt,'SubSystem')
                    srcRef = 'built-in/SubSystem';
                else
                    srcRef = ['built-in/' bt];
                end
            end

            % position + params
            pos = get_param(b,'Position');
            dp  = get_param(b,'DialogParameters');
            params = struct();
            params.Position = pos;
            if isstruct(dp)
                f = fieldnames(dp);
                for i = 1:numel(f)
                    prm = f{i};
                    try
                        v = get_param(b, prm);
                        if isnumeric(v) || islogical(v) || ischar(v) || isstring(v)
                            params.(prm) = v;
                        end
                    catch
                    end
                end
            end

            % add_block
            cmd = sprintf('add_block(%s,%s', q(srcRef), q(tgtPath));
            pnames = fieldnames(params);
            for i = 1:numel(pnames)
                prm = pnames{i};
                cmd = sprintf('%s,''%s'',%s', cmd, prm, lit(params.(prm)));
            end
            cmd = [cmd ');'];
            out = [out emit(cmd)];

            % recurse into subsystems
            if strcmpi(get_param(b,'BlockType'),'SubSystem')
                out = [out emit(emitSystem(b, tgtPath))];
            end
        end

        % --- lines (this level) ---
        ln = find_system(srcSystem,'FindAll','on','Type','Line');
        for h = ln(:).'
            try
                sp = get_param(h,'SrcPortHandle');
                dp = get_param(h,'DstPortHandle');
                if sp == -1 || isempty(dp), continue; end
                sblk = get_param(get_param(sp,'Parent'),'Name');
                sport= get_param(sp,'PortNumber');
                for dph = dp(:).'
                    if dph == -1, continue; end
                    dblk = get_param(get_param(dph,'Parent'),'Name');
                    dport= get_param(dph,'PortNumber');
                    srcEnd = sprintf('%s/%d', sblk, sport);
                    dstEnd = sprintf('%s/%d', dblk, dport);
                    out = [out emit(sprintf('add_line(%s,%s,%s,''autorouting'',''on'');', ...
                                q(tgtSystem), q(srcEnd), q(dstEnd)))];
                end
            catch
            end
        end
    end

    % quote helper
    function s = q(x)
        if isstring(x), x = char(x); end
        if ~ischar(x), x = char(string(x)); end
        x = strrep(x, '''', '''''');          % escape single quotes
        s = ['''' x ''''];
    end

    % literal helper
    function s = lit(v)
        if isnumeric(v) || islogical(v)
            s = mat2str(v);
        else
            s = q(v);
        end
    end
end

txt = exportSimulinkToScript('simulink_model_original','simulink_model_original_rebuild');
fid = fopen('rebuild_simulink_model_original.m','w'); fwrite(fid, txt); fclose(fid);
run('rebuild_simulink_model_original.m');