%% ear sch process demo figure
% use subject 1, which is mesh_ind = 6
ind = 6;    %6;
load('ear_SCH_all_correctCenter.mat', 'ear_SCH_all', 'Mapping_all');
load('earCanal_index.mat', 'EarCanal_ind');
load('C_mat_cap_25_0.02_20order_real.mat','C_mat_real', 'f_regularCap', 'map_regularCap');

path = cd;
mesh_dir = dir([path,'\*.ply']);
% mesh_dir = natsortfiles(mesh_dir);

[Tri,Pts] = plyread(mesh_dir(ind).name,'tri');

fv_map_correct = Mapping_all(ind).mapping;

map_corrected = fv_map_correct.vertices;

%% pick ear mesh area

    r_candidate = deg2rad(30);     %for the cap candidate
    
    %left ear
    leftEarCanal_ind = EarCanal_ind(ind ,1);
    
    center_l = zeros(1,2);
    [center_l(1), center_l(2)] = cart2sph(map_corrected(leftEarCanal_ind, 1), map_corrected(leftEarCanal_ind, 2), map_corrected(leftEarCanal_ind, 3) );
    
    center_l_originalSpace = zeros(1,2);
    [center_l_originalSpace(1), center_l_originalSpace(2)] = cart2sph(Pts(leftEarCanal_ind, 1), Pts(leftEarCanal_ind, 2), Pts(leftEarCanal_ind, 3) );

    map_leftEarUp = coord_rotation_top(map_corrected, rad2deg( center_l(1)), rad2deg( center_l(2)) );
    [TH_mapLeftEarUp, PHI_mapLeftEarUp] = cart2sph(map_leftEarUp(:,1), map_leftEarUp(:,2), map_leftEarUp(:,3));
    leftEar_ind = find(PHI_mapLeftEarUp > pi/2- r_candidate);
    
    Pts_leftEarUp = coord_rotation_top(Pts, rad2deg( center_l_originalSpace(1)), rad2deg( center_l_originalSpace(2)));
    
    ch_leftEar = convhulln([map_leftEarUp(leftEar_ind,:); [0,0,0]]);
    
%     test = [map_leftEarUp(leftEar_ind,:); [0,0,0]];
    
    fv_leftEar.faces = ch_leftEar;
    fv_leftEar.vertices = [Pts_leftEarUp(leftEar_ind,:); [0,0,0]];
    
    %% DO another resample here, with regular cap grid, from 30 degree into 25 degree
  
    
    % left ear
    map_in_left = map_leftEarUp(leftEar_ind,:);
  
    %%
    fv_leftEar_remesh = Mapping_all(ind).leftEarRemesh;
    
    %%
    qm_k_leftEar = squeeze(ear_SCH_all(ind, 1, :,:));
    v_rec = C_mat_real * qm_k_leftEar;  
%%
figure; subplot(221); patch('vertices', map_in_left, 'faces', convhulln(map_leftEarUp(leftEar_ind,:)), 'FaceColor', 'w', 'edgecolor','k'); axis equal tight
view(90,90)
title('left ear conformal mapping on sph cap of 30 degree')

subplot(222);patch(fv_leftEar, 'FaceColor', 'w', 'edgecolor','k'); axis equal tight
view(90,90)
xl = xlim;
yl = ylim;
zl = zlim;
title('left ear actual mesh area')

subplot(223); patch(fv_leftEar_remesh, 'FaceColor', 'w', 'edgecolor','k'); axis equal tight
view(90,90)
title('left ear remesh with uniform sph cap sampling')
xlim(xl);
ylim(yl);
zlim(zl);


subplot(224); patch('vertices', real(v_rec), 'faces', f_regularCap, 'FaceColor', 'w', 'edgecolor','k'); axis equal tight
view(90,90)
title('SCHA reconstruction, order 20')
xlim(xl);
ylim(yl);
zlim(zl);

%% optional, plot the title in paper way
% % subplot(221);
% % ttl = title('A','fontsize', 16);
% % ttl.Units = 'Normalize';
% % ttl.Position(1) = -.1; % use negative values (ie, -0.1) to move further left
% % ttl.HorizontalAlignment = 'left';
% % subplot(222);
% % ttl = title('B','fontsize', 16);
% % ttl.Units = 'Normalize';
% % ttl.Position(1) = -.1; % use negative values (ie, -0.1) to move further left
% % ttl.HorizontalAlignment = 'left';
% % subplot(223);
% % ttl = title('C','fontsize', 16);
% % ttl.Units = 'Normalize';
% % ttl.Position(1) = -.1; % use negative values (ie, -0.1) to move further left
% % ttl.HorizontalAlignment = 'left';
% % subplot(224);
% % ttl = title('D','fontsize', 16);
% % ttl.Units = 'Normalize';
% % ttl.Position(1) = -.1; % use negative values (ie, -0.1) to move further left
% % ttl.HorizontalAlignment = 'left';



