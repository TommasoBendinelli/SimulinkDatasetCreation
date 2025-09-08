classdef sld_FuelModes < Simulink.IntEnumType
% Fuel modes for sldemo_fuelsys

%
% Copyright 1994-2010 The MathWorks, Inc.

    enumeration
        LOW(1)
        RICH(2)
        DISABLED(3)
    end
end
