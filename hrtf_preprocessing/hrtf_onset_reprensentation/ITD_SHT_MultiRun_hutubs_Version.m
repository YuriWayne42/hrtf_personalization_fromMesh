 
% HUTUBS database contain 440 points, convering all spatial directions
% no need for conformal mapping
 
 

%f = [f(phi1,theta1),f(phi2,theta2)...]'
%c = [c00, c1-1, c10, c11...]'
%Y = [y00, y1-1, y10, y11...]  = SHbase_P
%ynm = [Ynm(phi1,theta1),...]'

% C_matrix is conbination of c, for 128 freqs
tstart = tic;
 
path = 'HRIRs\';
SH_order = 5;
input_hrir_onset = zeros(96, 440, 2);

HRTF_onset_SHT = zeros(96, 36, 2);
% hrtf_SHT_dBmat = zeros(96, 128, (SH_order+1)^2, 2);

% hrtf_dir = dir([path,'*measured.sofa'])

hrtf_dir = dir([path,'*measured.sofa']);

hrtf_dir = natsortfiles(hrtf_dir);
 

str = 'running';
hwait=waitbar(0,str);



for ind = 1:length(hrtf_dir)
    waitbar( ind/length(hrtf_dir) , hwait,str);
    
    hrtfData = SOFAload(strcat(path,'\', hrtf_dir(ind).name)  );
    
N = size(hrtfData.SourcePosition, 1);

input_locations_sph = deg2rad(hrtfData.SourcePosition(:,1:2));

TH = input_locations_sph(:,1);
PHI = input_locations_sph(:,2);
 

epsilon = 1e-6;               %
% phi_sample = (sqrt(5)-1)/2;

% 
% hrtf_freq_l = fft( squeeze(hrtfData.Data.IR(:,1,:) )' );
% hrtf_freq_r = fft( squeeze(hrtfData.Data.IR(:,2,:) )' );
% 
% hrtf_freq_l = hrtf_freq_l';
% hrtf_freq_r = hrtf_freq_r';
% str = 'running onset detection';
% hwait=waitbar(0,str);

for i = 1:length(TH)
%     waitbar(i/length(TH), hwait,str);
    
    
    input_hrir_onset(ind, i,1) = onset_detect(squeeze(hrtfData.Data.IR(i,1, 11:60) ), -20) + 10;     %usually -37?
    input_hrir_onset(ind, i,2) = onset_detect(squeeze(hrtfData.Data.IR(i,2, 11:60) ), -20) + 10;
    
end






end
%% 
path = 'onset_analysis\';

for ind = 1:length(hrtf_dir)
    h = figure(1);
    plot(squeeze(input_hrir_onset(ind, :,1))); hold on; plot(squeeze(input_hrir_onset(ind, :,2)))
    print(gcf,'-dpng', [path,num2str(ind),'_onset_validate.png'])
    close(h);
    
end

%%
tstart = tic;

str = 'running';
hwait=waitbar(0,str);
for ind = 1:length(hrtf_dir)
    waitbar( ind/length(hrtf_dir) , hwait,str);
    
    hrtfData = SOFAload(strcat(path,'\', hrtf_dir(ind).name)  );
    N = size(hrtfData.SourcePosition, 1);

input_locations_sph = deg2rad(hrtfData.SourcePosition(:,1:2));

TH = input_locations_sph(:,1);
PHI = input_locations_sph(:,2);



SHbase_P = 0;
SHbase_P(1:N,1:(SH_order+1)^2) = 0;
SHbase_P(1:N,1) = 1;   %0 order is 1 anyway


for i = 1:N
    SH_Vec = SHCreateYVec(SH_order, TH(i), pi/2 - PHI(i));
    SHbase_P (i, :) = SH_Vec';
    %     SHbase_P(i ,n^2+k+n+1) = PP(Sample_coor(i,2), Sample_coor(i,1));
end

SHbase_P = roundn(SHbase_P, -5);


f1 = squeeze(input_hrir_onset(ind, :,1));
f2 = squeeze(input_hrir_onset(ind, :,2));

f1 = f1(:);
f2 = f2(:);

%     C1 = [];
%     C2 = [];
% C = (SHbase_P.'*SHbase_P)\(SHbase_P.'*f);   %calc the weighting coeff

%testing on the Tikhonov Regulization
C1 = (SHbase_P.'*SHbase_P + epsilon* eye((SH_order+1)^2))\(SHbase_P.'*f1);
C2 = (SHbase_P.'*SHbase_P + epsilon* eye((SH_order+1)^2))\(SHbase_P.'*f2);

%     C_matrix_l(:,i_freq) = C1;
%     C_matrix_r(:,i_freq) = C2;


%     hrtf_SHT_mat = zeros(48, 128, (SH_order+1)^2, 2);

HRTF_onset_SHT(ind, :, 1) = C1;
HRTF_onset_SHT(ind, :, 2) = C2;


end

 

telapsed = toc(tstart) 

 