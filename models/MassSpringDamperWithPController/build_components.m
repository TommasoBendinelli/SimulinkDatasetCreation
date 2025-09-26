bdclose('all');

% Clean stale artifacts
if exist('slprj','dir'), rmdir('slprj','s'); end
delete('MyLib_lib.slx'); delete('MyLib_lib.slxc');

% (Optional) slim the path to avoid conflicts, then add only this folder
origPath = path; cleanupObj = onCleanup(@() path(origPath));
restoredefaultpath; rehash toolboxcache; addpath(pwd);

% Quick checks
disp(exist('+MyLib','dir'));     % should print 7
dir('+MyLib')                    % should list force_non_ideal.ssc

% Build
ssc_build('MyLib');
sl_refresh_customizations;
open_system('MyLib_lib');