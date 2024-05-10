function h = plotSphFunctionTriangle_edited(F, dirs)
%PLOTSPHFUNCTIONTRIANGLE Plots a spherical function on unstructured grid
%
%   F:  vector of K function values on the sampling points
%   dirs:   [azimuth1 elevation; ...; azimuthK elevation] angles in 
%           rads for each evaluation point 

[x,y,z] = sph2cart(dirs(:,1) , dirs(:,2) , abs(F) );



[x_unit,y_unit,z_unit] = sph2cart(dirs(:,1) , dirs(:,2) , ones(length(dirs),1) );

[ch] = convhulln([x_unit,y_unit,z_unit] );

% % plot 3d axes
% maxF = max(max(abs(F)));
% line([0 1.5*maxF],[0 0],[0 0],'color',[1 0 0])
% line([0 0],[0 1.5*maxF],[0 0],'color',[0 1 0])
% line([0 0],[0 0],[0 1.5*maxF],'color',[0 0 1])

% plot function
% h = trisurf(ch, x, y, z, F);

% h = trisurf(ch, x, y, z, F, 'edgecolor', 'none');
h = trisurf(ch, x, y, z, F, 'edgecolor', [.3 .3 .3]);

% h = trisurf(ch, x_unit,y_unit,z_unit, F ,'edgecolor', 'none');

% patch('vertices',[x_unit,y_unit,z_unit,], 'faces', ch,  'edgecolor','w');

% grid
axis equal
end
