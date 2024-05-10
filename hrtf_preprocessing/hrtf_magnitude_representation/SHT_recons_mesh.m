% plot 
function h = SHT_recons_mesh(C)
SH_order = sqrt(length(C)) - 1;

mesh_recreate = C(1)* SHbase(0,0);

for n = 1:SH_order
    for k = n*(-1):n
        mesh_recreate = mesh_recreate + C(SHlm2n(n,k))*SHbase(n,k);
        
    end
end
mesh_recreate = flipud(mesh_recreate);

% right now sampling cover the whole sphere
theta = linspace(0, pi, 181);
phi = linspace(0, 2*pi, 361);
[tt,pp]=meshgrid(theta,phi);



[x,y,z] = sph2cart(pp,pi/2-tt,abs(mesh_recreate));

h = surf(x,y,z, mesh_recreate); shading interp;
axis equal; material DULL

end