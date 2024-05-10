% only suitable for HUTUBS

% follow the method in "HRTF MAGNITUDE MODELING USING A NON-REGULARIZED" Jens Ahrens 
% construct lower order SHT according to existing cap, and obtain
% supporting point at lower elevation.

% in this version, delete src directions below certain elevation to check
% the SHT coeff consistancy

 

% HUTUBS database contain 440 points, convering all spatial directions
% no need for conformal mapping
% still using dB scale


%generate the least square fit for the HUTUBS HRTF and recreate it
% with all the points in the data

%f = [f(phi1,theta1),f(phi2,theta2)...]'
%c = [c00, c1-1, c10, c11...]'
%Y = [y00, y1-1, y10, y11...]  = SHbase_P
%ynm = [Ynm(phi1,theta1),...]'
[input_mat_name, pathname] = uigetfile('*.sofa','Pick an hrtf file in sofa');
hrtfData = SOFAload(input_mat_name);
 
% %HUTUBS uses spherical source locations in beginning
%now turn uniformly as sph
if strcmp(hrtfData.SourcePosition_Type,'spherical')
    SP=hrtfData.SourcePosition(:,:);
else
    SP  = SOFAconvertCoordinates(hrtfData.SourcePosition(:,:),hrtfData.SourcePosition_Type,'spherical');
end
% SP is the spherical coord of the src locations.

% [TH, PHI] = cart2sph(SP(:,1),SP(:,2),SP(:,3) );

input_locations_sph = deg2rad(SP(:,1:2));
TH = input_locations_sph(:,1);
PHI = input_locations_sph(:,2);


N = length(PHI);
SH_order = 10;
SH_order_lower = 3;
freq_ind = 42;


%% create SHT base with complete grid, 7 order

SHbase_P = [];
SHbase_P(1:N,1:(SH_order+1)^2) = 0; 
SHbase_P(1:N,1) = 1;   %0 order is 1 anyway


for i = 1:N    %parfor
    SH_Vec = SHCreateYVec(SH_order, TH(i), pi/2 - PHI(i));
    SHbase_P (i, :) = SH_Vec'; 
%     SHbase_P(i ,n^2+k+n+1) = PP(Sample_coor(i,2), Sample_coor(i,1)); 
end



%% discard lower elevations
% lowElev_ind = find(PHI(:) <= deg2rad(-60));
lowElev_ind = find(PHI(:) < deg2rad(-30));



TH_truncated = TH;
TH_truncated(lowElev_ind) = [];

PHI_truncated = PHI;
PHI_truncated(lowElev_ind) = [];


% check left hrtf just for a demo
input_hrtf = fft( squeeze(hrtfData.Data.IR(:,1,:) )' );
input_hrtf = input_hrtf';

% input_hrtf_mag = angle(input_hrtf); 
input_hrtf_mag = abs(input_hrtf); 

tstart = tic;
 
 
% SH_order = 7;
 

N = length(TH);
freq_vec = linspace(0, hrtfData.Data.SamplingRate/2, hrtfData.API.N./2+1);

%% select a single frequency hrtf 

% hrtfMag_singleF = abs(input_hrtf_mag(:,freq_ind));

hrtfMag_singleF = input_hrtf_mag(:,freq_ind);

hrtfMag_singleF = hrtfMag_singleF(:);

% hrtfMag_singleF = mag2db(hrtfMag_singleF.*1e5);

hrtfMag_singleF_truncated = hrtfMag_singleF;
hrtfMag_singleF_truncated(lowElev_ind) = [];

% hrtfMag_singleF = unwrap(hrtfMag_singleF);
% % hrtfMag_singleF = mag2db(hrtfMag_singleF ./ 20e-6 ); %./ 20e-6
% hrtfMag_singleF = (hrtfMag_singleF.^(2/3.16) );

%% assign f and plot it  
f_truncated = hrtfMag_singleF_truncated;
f_truncated = f_truncated(:);



% f = zeros(length(TH),1);
% f(32) = 1;

figure; 
subplot(241); 
plotSphFunctionTriangle_edited(f_truncated, [TH_truncated, PHI_truncated]);

cmap = getPyPlot_cMap('RdBu_r', 128); colormap(cmap)
 
% axis equal;
title(strcat( 'Original hrtf radial pattern, freq =', num2str( freq_vec(freq_ind)) )); %'FontSize',14, 'FontName', 'Arial'
 

%% compute SHT, and plot recreated mesh  
 
%testing on the Tikhonov Regulization
% C = (SHbase_P.'*SHbase_P + epsilon* eye((SH_order+1)^2))\(SHbase_P.'*f);
[C , f_recons] = SHT_core(f_truncated, [TH_truncated, PHI_truncated], SH_order);

subplot(242);
SHT_recons_mesh(C);
title(strcat('Recons shape without processing, SH order = ',num2str(SH_order))); %'FontSize',14, 'FontName', 'Arial'

%% plot original v. recons  

subplot(243); plot(f_truncated); 
hold on; plot(f_recons);
error_rms = rms(f_truncated - f_recons);
legend('original','reconstructed','location','best')

title(strcat('sample number = ',num2str(length(f_truncated)), ', RMS error =',num2str(error_rms)))

 
%% plot SHT coeff  
subplot(244); plot(C); title('weights of each SH coeff')

%% perform SHT, 3rd order, with incomplete grid
[C_lowerOrder , ~] = SHT_core(f_truncated, [TH_truncated, PHI_truncated], SH_order_lower);

%% recreate signal at complete grid(support point) with 3rd order SHT recons
f_C_lowerOrderRecons = SHbase_P(:, 1:(SH_order_lower+1)^2) * C_lowerOrder;

f_lowerOrderSupport = hrtfMag_singleF;
f_lowerOrderSupport(lowElev_ind) = f_C_lowerOrderRecons(lowElev_ind);

subplot(245);
plotSphFunctionTriangle_edited(f_lowerOrderSupport, [TH, PHI]);
colormap(cmap);
title('Pattern with 3rd order SHT recreated signal at support points')

%% perform SHT to f_lowerOrderSupport
[C_lowerOrderSupport , f_recons_method] = SHT_core(f_lowerOrderSupport, [TH, PHI], SH_order);

subplot(246);
SHT_recons_mesh(C_lowerOrderSupport);
title(strcat('Recons shape with processing, SH order = ',num2str(SH_order))); 


%% plot original v. recons  in new method
subplot(247); plot(f_lowerOrderSupport); 
hold on; plot(f_recons_method);
error_rms_method = rms(f_lowerOrderSupport - f_recons_method);
legend('original','reconstructed','location','best')

title(strcat('sample number = ',num2str(length(f_lowerOrderSupport)), ', RMS error =',num2str(error_rms_method)))

%% plot SHT coeff  in new method
subplot(248); plot(C_lowerOrderSupport); title('weights of each SH coeff, with processing')


