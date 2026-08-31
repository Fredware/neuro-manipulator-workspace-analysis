clear all
close all
clc

%% upperlimb_fig.m file must be placed in the same directory containing the struct files .mat

% Enter the code of the subject whose variables you want to view
% ST## : post-stroke patient (ex: ST03)
% HS## : healthy subject (ex: HS10)
subject = "HS07";

% Choose the variables you want to view (use the variable names in the
% structure, excluding _RX or _LX where present)
variable1 = "ELBOW";
variable2 = "BICEP";
variable3 = "ElbowFlx";

% Enter the respective code for each variable
% 0 : Marker Coordinates
% 1 : EMG
% 2 : Angle
code1 = 0;
code2 = 1;
code3 = 2;

variables = [variable1; variable2; variable3];
codes = [code1 code2 code3];
marker_len = 19;
emg_len = 12;
angles_len = 19;
tasks_len = 6;
vec_tk = -5:1:0;
check = ["ScapRot";"ScapLAtBend";"PIPF2";"PIPF5";"MCPF2";"MCPF5";"TAP"];

if contains(subject, "ST")
    dataname = "DataULpleg";
    side = "HemiSide";
elseif contains(subject, "HS")
    dataname = "DataULdom";
    side = "DomSide";
end
fig = figure;
for var = 1:length(variables)
    load(strcat(subject, '.mat'));
    for tk = 1:tasks_len
        switch codes(var)
            case 0
                var_marker = strcat(variables(var), "_", s.(side));
                for l = 1:length(s.(dataname)(tk).MarkerVarName)
                    if(contains(s.(dataname)(tk).MarkerVarName(l),var_marker))
                        idx_marker = l*3;
                        break;
                    end
                end
                subplot(length(variables),tasks_len, 6*var+vec_tk(tk))
                plot([s.(dataname)(tk).Marker(idx_marker-2,:); s.(dataname)(tk).Marker(idx_marker-1,:); s.(dataname)(tk).Marker(idx_marker,:)]')
                legend("x", "y", "z")
                title(s.(dataname)(tk).TaskCode)
                xlabel("Frames (f = 125Hz)")
                ylabel(strcat(variables(var), " Marker Coordinates [mm]"))
            case 1
                for l = 1:length(s.(dataname)(tk).EmgVarName)
                    if(contains(s.(dataname)(tk).EmgVarName(l),variables(var)))
                        idx_emg = l;
                        break;
                    end
                end
                subplot(length(variables),tasks_len, 6*var+vec_tk(tk))
                plot(s.(dataname)(tk).EMG(idx_emg,:))
                title(s.(dataname)(tk).TaskCode)
                xlabel("Frames (f = 1000Hz)")
                ylabel(strcat(variables(var), " EMG signal [mV]"))
            case 2
                if(strcmp(s.(side),"LX") && any(contains(check,variables(var))))
                    sig = -1;
                else 
                    sig = 1;
                end
                for l = 1:length(s.(dataname)(tk).AngleVarName)
                    if(contains(s.(dataname)(tk).AngleVarName(l),variables(var)))
                        idx_angle = l;
                        break;
                    end
                end
                subplot(length(variables),tasks_len, 6*var+vec_tk(tk))
                plot(sig*s.(dataname)(tk).Angles(idx_angle,:))
                title(s.(dataname)(tk).TaskCode)
                xlabel("Frames (f = 125Hz)")
                ylabel(strcat(variables(var), " Angle [°]"))
        end       
    end
end
