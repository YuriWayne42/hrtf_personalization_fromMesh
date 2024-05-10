function h = SH_pyramidPlot(C)
% takes input C vector, plot the SH coeff in pyramid form

SH_order = sqrt(length(C)) - 1;

SH_pyramid = ones(SH_order+1, SH_order*2 +1).*(max(C)+min(C))./2;
for C_ind = 1:length(C)
    [l, m] = SHn2lm(C_ind);
    SH_pyramid(l+1 , SH_order +1 + m ) = C(C_ind);
    
end
l = 0: SH_order;
m = -SH_order : SH_order;
h = imagesc(m, l, SH_pyramid); 
axis equal;


ylim([-.5, SH_order+.5]);

xlabel('m value');
ylabel('l value');
colorbar;


end 