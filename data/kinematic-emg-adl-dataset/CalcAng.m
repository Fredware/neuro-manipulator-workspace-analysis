function [flessione,adduzione,rotazione] = CalcAng(ProxX,ProxY,ProxZ,DistX,DistY,DistZ)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
% This function calculates, for a single time instant, the angles of     % 
% flexion (flex), adduction (add) and rotation (rot) starting from       %
% two triads of axes                                                     %
%                                                                        %
%  Input:                                                                %
% - ProxX,ProxY,ProxZ: directional cosines of the reference frame of the %
%   proximal segment (with respect to an absolute reporting system).     %
% - DistX,DistY,DistZ: directional cosines of the reference frame of the %
%   distal segment (with respect to an absolute reporting system).       %
% Output:                                                                %
% - flex: flexion angle in the sagittal plane                            %
% - add: adduction angle in frontal plane                                %
% - rot: rotation angle in the transverse plane                          %
%                                                                        %
% Grood ES, Suntay WJ. A joint coordinate system for the clinical        %
% description of three-dimensional motions: application to the knee.     %
% J Biomech Eng. 1983;105(2):136-144.                                    %
%                                                                        %
% Wu G, van der Helm FC, Veeger HE, Makhsous M, Van Roy P, Anglin C,     %
% Nagels J, Karduna AR, McQuade K, Wang X, Werner FW, Buchholz B;        %
% International Society of Biomechanics. ISB recommendation on           %
% definitions of joint coordinate systems of various joints for the      %
% reporting of human joint motion--Part II: shoulder, elbow, wrist and   %
% hand. J Biomech. 2005 May;38(5):981-992.                               %
% doi: 10.1016/j.jbiomech.2004.05.042. PMID: 15844264.                   %
%                                                                        %
% Copyright Marco Rabuffetti                                             %
%                                                                        %                       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nodi = cross(DistY,ProxZ);
nodi = nodi/norm(nodi);
adduzione = pi/2 - acos(dot(ProxZ,DistY));
rotazione = pi/2 - acos(dot(nodi, DistZ));
if dot(cross(nodi,ProxY),ProxZ)>0,
    flessione = pi/2 - acos(dot(nodi, ProxY));
else
    flessione = pi/2 - (2*pi - acos(dot(nodi, ProxY)));
end