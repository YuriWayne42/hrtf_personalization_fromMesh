% manually 
% load('mesh_SHT_all.mat', 'mesh_ind')
load('earCanal_index.mat', 'EarCanal_ind');
load('C_mat_cap_25_0.02_20order_real.mat','C_mat_real', 'f_regularCap', 'map_regularCap');
    

path = cd;
mesh_dir = dir([path,'\*.ply']);

max_degree = 20;

ear_SCH_all = zeros(length(mesh_dir), 2, (max_degree+1)^2, 3);
Mapping_all = [];
Mapping_all.mapping = [];
Mapping_all.leftEarRemesh = [];
Mapping_all.rightEarRemesh = [];

tstart = tic;

str = 'running';
hwait = waitbar(0,str);

for ind = 1:length(mesh_dir)
    waitbar( ind/length(mesh_dir) , hwait,str);
    
    %% load mesh
    [Tri,Pts] = plyread(mesh_dir(ind).name,'tri');
    
    [TH, PHI] = cart2sph(Pts(:,1), Pts(:,2), Pts(:,3));
    
    % find the most frontal/upward point index, use them to correct map
    [~, front_tipInd] = min(vecnorm([TH, PHI]'));
    [~, up_tipInd] = max(PHI);
    
    %% paramaterization step, conformal map, with mobius correction
    map = spherical_conformal_map(Pts, Tri);
    [map_mobius, x] =  mobius_area_correction_spherical(Pts, Tri, map);
    
    map = map_mobius;
    
    ch = convhulln(map);
    %     fv_head.faces = ch;
    %     fv_head.vertices = Pts;
    %
    %         fv_map.faces = ch;
    %         fv_map.vertices = map;
    
    
    %now make a correction, let map(up_tipInd) to be up, and map(front_tipInd)
    %to be front again.
    
    % first rotate
    [azi_uptip, elev_uptip] = cart2sph( map(up_tipInd,1), map(up_tipInd,2), map(up_tipInd,3) );
    map_upped = coord_rotation_top(map, rad2deg(azi_uptip), rad2deg(elev_uptip));
    
    [azi_fronttip, ~] = cart2sph( map_upped(front_tipInd,1), map_upped(front_tipInd,2), map_upped(front_tipInd,3) );
    map_corrected = coord_rotation_top(map_upped, rad2deg(azi_fronttip), 90);
    
        fv_map_correct.faces = ch;
        fv_map_correct.vertices = map_corrected;
    
    %% pick ear mesh area
    r_candidate = deg2rad(30);     %for the cap candidate
    
    %left ear
    leftEarCanal_ind = EarCanal_ind(ind ,1);
    rightEarCanal_ind = EarCanal_ind(ind ,2);
    
    center_l = zeros(1,2);
    [center_l(1), center_l(2)] = cart2sph(map_corrected(leftEarCanal_ind, 1), map_corrected(leftEarCanal_ind, 2), map_corrected(leftEarCanal_ind, 3) );
    
    center_l_originalSpace = zeros(1,2);
    [center_l_originalSpace(1), center_l_originalSpace(2)] = cart2sph(Pts(leftEarCanal_ind, 1), Pts(leftEarCanal_ind, 2), Pts(leftEarCanal_ind, 3) );

    
    map_leftEarUp = coord_rotation_top(map_corrected, rad2deg( center_l(1)), rad2deg( center_l(2)) );
    [TH_mapLeftEarUp, PHI_mapLeftEarUp] = cart2sph(map_leftEarUp(:,1), map_leftEarUp(:,2), map_leftEarUp(:,3));
    leftEar_ind = find(PHI_mapLeftEarUp > pi/2- r_candidate);
    
    Pts_leftEarUp = coord_rotation_top(Pts, rad2deg( center_l_originalSpace(1)), rad2deg( center_l_originalSpace(2)) );
    
    ch_leftEar = convhulln(map_leftEarUp(leftEar_ind,:));
    
    fv_leftEar.faces = ch_leftEar;
    fv_leftEar.vertices = Pts_leftEarUp(leftEar_ind,:);
    
    
    %right ear
    center_r = zeros(1,2);
    [center_r(1), center_r(2)] = cart2sph(map_corrected(rightEarCanal_ind, 1), map_corrected(rightEarCanal_ind, 2), map_corrected(rightEarCanal_ind, 3) );
    
    center_r_originalSpace = zeros(1,2);
    [center_r_originalSpace(1), center_r_originalSpace(2)] = cart2sph(Pts(rightEarCanal_ind, 1), Pts(rightEarCanal_ind, 2), Pts(rightEarCanal_ind, 3) );

    
    map_rightEarUp = coord_rotation_top(map_corrected, rad2deg( center_r(1)), rad2deg( center_r(2)) );
    [TH_mapRightEarUp, PHI_mapRightEarUp] = cart2sph(map_rightEarUp(:,1), map_rightEarUp(:,2), map_rightEarUp(:,3));
    rightEar_ind = find(PHI_mapRightEarUp > pi/2- r_candidate);
    
    Pts_rightEarUp = coord_rotation_top(Pts, rad2deg( center_r_originalSpace(1)), rad2deg( center_r_originalSpace(2)) );
    
    ch_rightEar = convhulln(map_rightEarUp(rightEar_ind,:));
    
    fv_rightEar.faces = ch_rightEar;
    fv_rightEar.vertices = Pts_rightEarUp(rightEar_ind,:);
    
    %% DO another resample here, with regular cap grid, from 30 degree into 25 degree
    % edge_length = .02;%.032;
    % resolution = 60;
    r = deg2rad(25);     %for the cap theta_c
    % [map_regularCap, f_regularCap] = uniform_spherical_cap_grid(r, edge_length, resolution, 1);
    
    % left ear
    map_in_left = map_leftEarUp(leftEar_ind,:);
    
    leftEar_remesh_x =  my_remesh(Pts_leftEarUp(leftEar_ind,1), map_in_left, map_regularCap);
    leftEar_remesh_y =  my_remesh(Pts_leftEarUp(leftEar_ind,2), map_in_left, map_regularCap);
    leftEar_remesh_z =  my_remesh(Pts_leftEarUp(leftEar_ind,3), map_in_left, map_regularCap);
    
    fv_leftEar_remesh.faces = f_regularCap;
    fv_leftEar_remesh.vertices = [leftEar_remesh_x, leftEar_remesh_y, leftEar_remesh_z];
    
    
    % right ear
    map_in_right = map_rightEarUp(rightEar_ind,:);
    
    rightEar_remesh_x =  my_remesh(Pts_rightEarUp(rightEar_ind,1), map_in_right, map_regularCap);
    rightEar_remesh_y =  my_remesh(Pts_rightEarUp(rightEar_ind,2), map_in_right, map_regularCap);
    rightEar_remesh_z =  my_remesh(Pts_rightEarUp(rightEar_ind,3), map_in_right, map_regularCap);
    
    fv_rightEar_remesh.faces = f_regularCap;
    fv_rightEar_remesh.vertices = [rightEar_remesh_x, rightEar_remesh_y, rightEar_remesh_z];
    
    %% now do SCHA to the resampled ears
    qm_k_leftEar = (C_mat_real.' * C_mat_real) \ C_mat_real.' * fv_leftEar_remesh.vertices;

    qm_k_rightEar = (C_mat_real.' * C_mat_real) \ C_mat_real.' * fv_rightEar_remesh.vertices;

    ear_SCH_all(ind, 1, :,:) = qm_k_leftEar;
    ear_SCH_all(ind, 2, :,:) = qm_k_rightEar;
    
    
    Mapping_all(ind).mapping = fv_map_correct;
    Mapping_all(ind).leftEarRemesh = fv_leftEar_remesh;
    Mapping_all(ind).rightEarRemesh = fv_rightEar_remesh;
    
    
    
end

telapsed = toc(tstart);

% save('ear_SCH_all.mat', 'ear_SCH_all', 'Mapping_all');
