function Ima_Range = Fun_dense_range_map(calib,calib_dir,base_dir,...
    frame,ImaRGB)

T = Fun_open_calib(calib.name,calib_dir);
% T = Fun_open_calib(calib(frame+1).name,calib_dir);%
% ������Ҽӵģ�Ϊ���ǿ��Ե�������Ĵ��룬��ȡ��frame���״��ļ�

fd = fopen(sprintf('%s/%06d.bin',base_dir,frame),'rb');% ���״�������ļ�
if fd < 1
    fprintf('No LIDAR files !!!\n');
    keyboard
else
velo = fread(fd,[4 inf],'single')';% ��ȡ�״����ݣ���Ϊn*4�ľ���
%figure;plot3(velo(:,1),velo(:,2),velo(:,3),'.');xlabel('X');ylabel('Y');zlabel('Z');title('Original Velo');
% ���ﻭͼ�ĳ������Ҽӵģ�Ϊ�˿��м���
fclose(fd);
end

% remove all points behind image plane (approximation)
idx = velo(:,1)<0;% ���Ե�һ����С��5�ĵ㣬ΪʲôȡֵΪ5����Ҳ��֪����why 5?
velo(idx,:) = [];
% figure;plot3(px(:,1),px(:,2),px(:,3),'.');% ���ﻭͼ�ĳ������Ҽӵģ�Ϊ�˿��м���
%ppx=velo;
% project to image plane (exclude luminance)
px = (T.P2 * T.R0_rect * T.Tr_velo_to_cam * velo')';% ��ת��RGBƽ�棬��ʽ�ο�KITTI��������У���ļ���ʹ�õ�readme
px(:,1) = px(:,1)./px(:,3);
px(:,2) = px(:,2)./px(:,3);
ppx=px;
%figure;plot3(px(:,1),px(:,2),px(:,3),'.');xlabel('X');ylabel('Y');zlabel('Z');title('project to image plane');
% ���ﻭͼ�ĳ������Ҽӵģ�Ϊ�˿��м���

% % -----------------------------------------------------------------------
ix = px(:,1)<1;                 px(ix,:)=[];velo(ix,:)=[];% ����λ��С��1������
ix = px(:,1)>size(ImaRGB,2);    px(ix,:)=[];velo(ix,:)=[];% ���Գ���RGB��Χ������   
%  figure;% ���ﻭͼ�ĳ������Ҽӵģ�Ϊ�˿��м���
%  m3=max(px(:,3));collor=[px(:,3)/m3 px(:,3)/m3 px(:,3)/m3];
%  for i=1:length(collor)
%      hold on;
%      plot3(px(i,1),px(i,2),px(i,3),'.','color',collor(i,:));
%  end
%  xlabel('X');ylabel('Y');zlabel('Z');
% % Ordering
Pts = zeros(size(px,1),4);
Pts = sortrows(px,2);% ���յڶ�������
% % ======================= Interpolation / Upsampling :::
c_px = floor(min(px(:,2)));% �Եڶ�����С��������ȡ֤������С������λ��
i_size = size(ImaRGB(c_px:end,:,1));% 
Ima3D = zeros( size(ImaRGB(:,:,1)) );% ��ʼ��һ�����������洢���ܻ������ͼ

% Simply type: mex fun_dense3D.cpp
Ima3D(c_px:end,:) = fun_dense3D(Pts,[c_px i_size]); % MEX-file ����C++����
% % -----------------------------------------------------------------------
% Normalization 8 bits

idxx=find(Ima3D==0); % find all 1.5
Ima3D(idxx)=max(max(Ima3D)); % set 1 to these indexes
Ima_Range = uint8( 255*Ima3D/max(max(Ima3D)) ); % :)% תΪuint8��ʽ���õ����ճ��ܻ������ͼ
%��ʾ�״�ɢ��ͼ��image
figure(1);
%subplot(2,1,1);
image(ImaRGB);
coord=fix(Pts(:,1:3));
%m3=max(ppx(:,3));
%collor=[ppx(:,3)/max(ppx(:,3)) ppx(:,3)/max(ppx(:,3)) ppx(:,3)/max(ppx(:,3)) ppx(:,3)/max(ppx(:,3))];
%coordd=[sqrt(coord(:,1).*coord(:,1)+coord(:,2).*coord(:,2)) sqrt(coord(:,1).*coord(:,1)+coord(:,2).*coord(:,2)) sqrt(coord(:,1).*coord(:,1)+coord(:,2).*coord(:,2))];
velo=abs(velo);
collor=[velo(:,1)/max(velo(:,1)) velo(:,2)/max(velo(:,2)) velo(:,3)/max(velo(:,3))];
hold on;
%scatter(coord(:,1),coord(:,2),1,'red');
%subplot(2,1,1);
scatter(coord(:,1),coord(:,2),3,collor);

end %END main Function
