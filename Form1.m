function varargout = Form1(varargin)
% FORM1 MATLAB code for Form1.fig
%      FORM1, by itself, creates a new FORM1 or raises the existing
%      singleton*.
%
%      H = FORM1 returns the handle to a new FORM1 or the handle to
%      the existing singleton*.
%
%      FORM1('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in FORM1.M with the given input arguments.
%
%      FORM1('Property','Value',...) creates a new FORM1 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Form1_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Form1_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Form1

% Last Modified by GUIDE v2.5 23-Jul-2020 23:43:30

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Form1_OpeningFcn, ...
                   'gui_OutputFcn',  @Form1_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Form1 is made visible.
function Form1_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Form1 (see VARARGIN)

% Choose default command line output for Form1
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Form1 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Form1_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in dossec.
function dossec_Callback(hObject, eventdata, handles)
% hObject    handle to dossec (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[isim, yol]=uigetfile({'*.jpg';'*.gif';'*.png';'*.ppm';'*.pgm'}, 'Görüntü Dosyasini Seç:','Multiselect','Off');
image = fullfile(yol,isim);
image = imread(image);
setappdata(0,'image',image);
axes(handles.axes1);
imshow(image);
fileinfo = dir(fullfile(yol,isim));
SIZE = fileinfo.bytes;
Size = SIZE/1024;
set(handles.edit2,'string',Size);





% --- Executes on button press in reset.
function reset_Callback(hObject, eventdata, handles)
% hObject    handle to reset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
setappdata(0,'image',ones(1,1));
axes(handles.axes1);
imshow(ones(1,1));
axes(handles.axes2);
imshow(ones(1,1));

% --- Executes on button press in doskay.
function doskay_Callback(hObject, eventdata, handles)
% hObject    handle to doskay (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%outimg = getappdata(0,'outimg');
F = getframe(handles.axes2);
Image = frame2im(F);
filter = { '*.jpg';'*.gif';'*.png';'*.ppm';'*.pgm';'*.bmp';'*.tiff';'* .m' };
[file, path] = uiputfile (filter);

if file == 0
  return;
end
fullFileName = fullfile(path, file);
imwrite(Image, fullFileName);


% --- Executes on button press in nokbazis.
function nokbazis_Callback(hObject, eventdata, handles)
% hObject    handle to nokbazis (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.donme,'visible','on');
set(handles.tab1,'visible','on');
set(handles.tab2,'visible','off');
set(handles.tab3,'visible','off');
set(handles.tab4,'visible','off');
set(handles.edit4,'visible','on');
set(handles.siki,'visible','off');
set(handles.edit1,'visible','off');


% --- Executes on button press in filtreis.
function filtreis_Callback(hObject, eventdata, handles)
% hObject    handle to filtreis (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.tab1,'visible','off');
set(handles.tab2,'visible','on');
set(handles.tab3,'visible','off');
set(handles.tab4,'visible','off');
set(handles.donme,'visible','off');
set(handles.edit4,'visible','off');
set(handles.siki,'visible','off');
set(handles.edit1,'visible','on');

% --- Executes on button press in renkui.
function renkui_Callback(hObject, eventdata, handles)
% hObject    handle to renkui (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.tab1,'visible','off');
set(handles.tab2,'visible','off');
set(handles.tab3,'visible','on');
set(handles.tab4,'visible','off');
set(handles.donme,'visible','off');
set(handles.edit4,'visible','off');
set(handles.siki,'visible','off');
set(handles.edit1,'visible','off');

% --- Executes on button press in si.
function si_Callback(hObject, eventdata, handles)
% hObject    handle to si (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.tab1,'visible','off');
set(handles.tab2,'visible','off');
set(handles.tab3,'visible','off');
set(handles.tab4,'visible','on');
set(handles.donme,'visible','off');
set(handles.edit4,'visible','off');
set(handles.siki,'visible','on');
set(handles.edit1,'visible','off');


% --- Executes on button press in rtoh.
function rtoh_Callback(hObject, eventdata, handles)
% hObject    handle to rtoh (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

resim= getappdata(0,'image');                                        
I=double(resim)/255;
R=I(:,:,1);
G=I(:,:,2);
B=I(:,:,3);

numi=1/2*((R-G)+(R-B));
denom=((R-G).^2+((R-B).*(G-B))).^0.5;

H=acosd(numi./(denom+0.000001));

H(B>G)=360-H(B>G);

H=H/360;

S=1- (3./(sum(I,3)+0.000001)).*min(I,[],3);

I=sum(I,3)./3;

HSI=zeros(size(resim));
HSI(:,:,1)=H;
HSI(:,:,2)=S;
HSI(:,:,3)=I;

figure,imshow(H);title('H Image');
figure,imshow(S);title('S Image');
figure,imshow(I);title('I Image');
setappdata(0,'HSI',HSI);
outimg=HSI;
setappdata(0,'outimg',outimg);
axes(handles.axes2);
imshow(outimg);

% --- Executes on button press in htor.
function htor_Callback(hObject, eventdata, handles)
% hObject    handle to grisev (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 HSI=getappdata(0,'HSI');
 setappdata(0,'image',HSI);
 axes(handles.axes1);
 imshow(HSI);

 H1=HSI(:,:,1);  
 S1=HSI(:,:,2);  
 I1=HSI(:,:,3);  
 
 H1=H1*360;                                               
   
 R1=zeros(size(H1));  
 G1=zeros(size(H1));  
 B1=zeros(size(H1));  
 RGB1=zeros([size(H1),3]);  

 B1(H1<120)=I1(H1<120).*(1-S1(H1<120));  
 R1(H1<120)=I1(H1<120).*(1+((S1(H1<120).*cosd(H1(H1<120)))./cosd(60-H1(H1<120))));  
 G1(H1<120)=3.*I1(H1<120)-(R1(H1<120)+B1(H1<120));  
 
 H2=H1-120;  

 R1(H1>=120&H1<240)=I1(H1>=120&H1<240).*(1-S1(H1>=120&H1<240));  
 G1(H1>=120&H1<240)=I1(H1>=120&H1<240).*(1+((S1(H1>=120&H1<240).*cosd(H2(H1>=120&H1<240)))./cosd(60-H2(H1>=120&H1<240))));  
 B1(H1>=120&H1<240)=3.*I1(H1>=120&H1<240)-(R1(H1>=120&H1<240)+G1(H1>=120&H1<240));  

 H2=H1-240;  

 G1(H1>=240&H1<=360)=I1(H1>=240&H1<=360).*(1-S1(H1>=240&H1<=360));  
 B1(H1>=240&H1<=360)=I1(H1>=240&H1<=360).*(1+((S1(H1>=240&H1<=360).*cosd(H2(H1>=240&H1<=360)))./cosd(60-H2(H1>=240&H1<=360))));  
 R1(H1>=240&H1<=360)=3.*I1(H1>=240&H1<=360)-(G1(H1>=240&H1<=360)+B1(H1>=240&H1<=360));  
 
 RGB1(:,:,1)=R1;  
 RGB1(:,:,2)=G1;  
 RGB1(:,:,3)=B1;  
 
 RGB1=im2uint8(RGB1);  
 figure,imshow(R1);title('R Image');  
 figure,imshow(G1);title('G Image');  
 figure,imshow(B1);title('B Image');  
 
outimg=RGB1;
setappdata(0,'outimg',outimg);
axes(handles.axes2);
imshow(outimg);

% --- Executes on button press in grisev.
function grisev_Callback(hObject, eventdata, handles)
% hObject    handle to grisev (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
res = getappdata(0,'image');
[str,stn,ktmn] = size(res);
if (ktmn > 1)
    R = res(:,:,1);
    G = res(:,:,2);
    B = res(:,:,3);
    res = .299*double(R) + .587*double(G) + .114*double(B);
    res = uint8(res);
end
outimg = res;
setappdata(0,'outimg',outimg);
axes(handles.axes2);
imshow(outimg);

% --- Executes on button press in haar.
function haar_Callback(hObject, eventdata, handles)
% hObject    handle to haar (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 bs=4; % block size
resim=getappdata(0,'image');
delta=0.01;
in=double(resim);
rgb=double(resim);
len=length(size(rgb));
for j1 = 1:3
t=rgb(:,:,j1);
n=bs;
Level=log2(n);
if 2^Level<n, error('block size should be 2,4,8,16');
end 
H=[1];
NC=1/sqrt(2);
LP=[1 1]; 
HP=[1 -1];
for i=1:Level
 H=NC*[kron(H,LP);kron(eye(size(H)),HP)];
end
H1=H;
H2=H;
H3=H;
H1=normc(H1);
H2=normc(H2);
H3=normc(H3);
H=H1*H2*H3;
x=t;
y=zeros(size(x));
[r,c]=size(x);
for i=0:bs:r-bs
 for j=0:bs:c-bs
 p=i+1;
 q=j+1;
 y(p:p+bs-1,q:q+bs-1)=(H')*x(p:p+bs-1,q:q+bs-1)*H;
 end
end
n1=nnz(y);
z=y;
m=max(max(y));
y=y/m;
y(abs(y)<delta)=0; %replace too low value to zero. 
y=y*m;
n2=nnz(y);
for i=0:bs:r-bs
 for j=0:bs:c-bs
 p=i+1;
 q=j+1;
 z(p:p+bs-1,q:q+bs-1)=H*y(p:p+bs-1,q:q+bs-1)*H';
 end
end
rgb(:,:,j1)=z;
end

imwrite(uint8(rgb),'haar_dalgacik.jpg');
boyut = dir('haar_dalgacik.jpg');
SIZE=boyut.bytes;
Size=SIZE/1024;
set(handles.edit3,'string',Size);

outimg=uint8(rgb);
setappdata(0,'outimg',outimg);
axes(handles.axes2);
imshow(uint8(rgb));


% --- Executes on button press in huffman.
function huffman_Callback(hObject, eventdata, handles)
% hObject    handle to huffman (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
resim = getappdata(0,'image');
h=Huffman(resim);
boyut = dir('huffman.jpg');
SIZE=boyut.bytes;
Size=SIZE/1024;
set(handles.edit3,'string',Size);

outimg=h;
setappdata(0,'outimg',outimg);
axes(handles.axes2);
imshow(h);

% --- Executes on button press in dct.
function dct_Callback(hObject, eventdata, handles)
% hObject    handle to dct (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
resim = getappdata(0,'image');

dct=DCT(resim);
boyut = dir('dct.jpg');
SIZE=boyut.bytes;
Size=SIZE/1024;
set(handles.edit3,'string',Size);

outimg=dct;
setappdata(0,'outimg',outimg);
axes(handles.axes2);
imshow(dct);

% --- Executes on button press in btc.
function btc_Callback(hObject, eventdata, handles)
% hObject    handle to btc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
resim=getappdata(0,'image');
if size(resim,3)==3
    resim=rgb2gray(resim);
end
%size of image
[M,N]=size(resim);
%convert to double
resim=double(resim);
Y=zeros(M,N);
blksize=2;    %Block Size
mu=colfilt(resim,[blksize,blksize],'distinct',@(x) ones(blksize^2,1)*mean(x));
sigma=colfilt(resim,[blksize,blksize],'distinct',@(x) ones(blksize^2,1)*std(x));
q=resim>mu;
q=colfilt(q,[blksize,blksize],'distinct',@(x) ones(blksize^2,1)*sum(x));
m=blksize^2;                         
a=mu-sigma.*(sqrt(q./m-q));           
b=mu+sigma.*(sqrt(m-q./q));           
H=resim>=mu;                              
Y(H)=a(H);
Y(~H)=b(~H);
Y=uint8(Y);                           
resim2=Y;
imwrite(resim2,'btc.jpg');
boyut = dir('btc.jpg');
SIZE=boyut.bytes;
Size=SIZE/1024;
set(handles.edit3,'string',Size);
outimg=Y;
setappdata(0,'outimg',outimg);
axes(handles.axes2);
imshow(Y);





% --- Executes on button press in mean.
function mean_Callback(hObject, eventdata, handles)
% hObject    handle to mean (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
fb = str2double(get(handles.edit1,'String'));
if (fb>0 && fb<4)
 
if(fb==2)
    fb=3;
end
resim = getappdata(0,'image');
filtreBoyutu=fb;
[row,col,katman]=size(resim);
if(katman>1)
    resim=rgb2gray(resim);
end
genisleme=(filtreBoyutu-1)/2+1;
baslama=((filtreBoyutu-1)/2);
resim2=zeros(row+genisleme,col+genisleme);  
[row1,col1]=size(resim2);
 

for i=2:row1-genisleme
    for j=2:col1-genisleme
        resim2(i,j)=resim(i,j);
    end
end
for y=2:row1-baslama
  for x=2:col1-baslama
       komsu=resim2((y-baslama):y+baslama,(x-baslama):x+baslama);
       resim2(y,x)=mean(komsu(:));
  end
end

outimg=uint8(resim2);
setappdata(0,'outimg',outimg);
axes(handles.axes2);
imshow(uint8(resim2));
else
    msgbox(' Girdiðiniz deger 0 dan büyük 4 den küçük olmalý');
end


% --- Executes on button press in gauss.
function gauss_Callback(hObject, eventdata, handles)
% hObject    handle to gauss (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
fb = str2double(get(handles.edit1,'String'));
if (fb>0 && fb<4)
 resim = getappdata(0,'image');

[~,~,katman]=size(resim);
if(katman>1)
   resim=rgb2gray(resim);
end
resimD = double(resim);
sigma=fb;
boyut = 4;

[X,Y]=meshgrid(-boyut:boyut,-boyut:boyut);

M = size(X,1)-1;

N = size(Y,1)-1;

Temp = -(X.^2+Y.^2)/(2*sigma*sigma);

hesap= exp(Temp)/(2*pi*sigma*sigma);

res=zeros(size(resimD));

resimD = padarray(resimD,[boyut boyut]);

for i = 1:size(resimD,1)-M
    
    for j =1:size(resimD,2)-N
        Temp = resimD(i:i+M,j:j+M).*hesap;
        
        res(i,j)=sum(Temp(:));
    end
    
end

outimg = uint8(res);
setappdata(0,'outimg',outimg);
axes(handles.axes2);
imshow(uint8(res));
else
  msgbox(' Girdiðiniz deger 0 dan büyük 4 den küçük olmalý');
end




% --- Executes on button press in sobel.
function sobel_Callback(hObject, eventdata, handles)
% hObject    handle to sobel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
fs = str2double(get(handles.edit1,'String'));
if (fs>0 && fs<4)
resim=getappdata(0,'image');
filtreBoyutu=fs;
[row,col,katman]=size(resim);
if(katman>1)
   resim=rgb2gray(resim);
end

boyut1=(filtreBoyutu-1)/2;
boyut2=((filtreBoyutu-1)/2)+1;
resim=double(resim);
resim2=double(zeros(row-filtreBoyutu-1,col-filtreBoyutu-1));    
filtre1=[-1 -2 -1 ; 0 0 0; 1  2 1];
filtre2=[-1 0 1 ; -2 0 2; -1 0 1];
for y=boyut2:row-boyut2
    for x=boyut2:col-boyut2
        komsu=resim((y-boyut1):y+boyut1,x-boyut1:x+boyut1);
        toplam1=0;
        toplam2=0;
        for m=1:filtreBoyutu
            for n=1:filtreBoyutu
                toplam1=toplam1+filtre1(m,n)*komsu(m,n);
            end
        end
        for m=1:filtreBoyutu
            for n=1:filtreBoyutu
                toplam2=toplam2+filtre2(m,n)*komsu(m,n);
            end
        end
        deger=power(toplam1,2)+power(toplam2,2);
        resim2(y-boyut1,x-boyut1)=sqrt(deger);
    end
end

outimg=uint8(resim2);
setappdata(0,'outimg',outimg);
axes(handles.axes2);
imshow(uint8(resim2)); 
else
    msgbox(' Girdiðiniz deger 0 dan büyük 4 den küçük olmalý');
end




% --- Executes on button press in prewitt.
function prewitt_Callback(hObject, eventdata, handles)
% hObject    handle to prewitt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
fs = str2double(get(handles.edit1,'String'));
if (fs>0 && fs<4)
 resim = getappdata(0,'image');

filtreBoyutu=fs;
[row,col,katman]=size(resim);
if(katman>1)
   resim=rgb2gray(resim);
end
boyut1=(filtreBoyutu-1)/2;
boyut2=((filtreBoyutu-1)/2)+1;
resim=double(resim);
resim2=double(zeros(row-filtreBoyutu-1,col-filtreBoyutu-1));    
filtre1=[-1 -1 -1; 0 0 0; 1 1 1];
filtre2=[-1 0 1; -1 0 1; -1 0 1];
for y=boyut2:row-boyut2
    for x=boyut2:col-boyut2
        komsu=resim((y-boyut1):y+boyut1,x-boyut1:x+boyut1);
        toplam1=0;
        toplam2=0;
        for m=1:filtreBoyutu
            for n=1:filtreBoyutu
                toplam1=toplam1+filtre1(m,n)*komsu(m,n);
            end
        end
        for m=1:filtreBoyutu
            for n=1:filtreBoyutu
                toplam2=toplam2+filtre2(m,n)*komsu(m,n);
            end
        end
        deger=power(toplam1,2)+power(toplam2,2);
        resim2(y-boyut1,x-boyut1)=sqrt(deger);
    end
end

outimg=uint8(resim2);
setappdata(0,'outimg',outimg);
axes(handles.axes2);
imshow(outimg);   
else
  msgbox(' Girdiðiniz deger 0 dan büyük 4 den küçük olmalý');
end


% --- Executes on button press in laplace.
function laplace_Callback(hObject, eventdata, handles)
% hObject    handle to laplace (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
fs = str2double(get(handles.edit1,'String'));
if (fs>0 && fs<4)
    resim = getappdata(0,'image');

filtreBoyutu=fs;
[row,col,katman]=size(resim);
if(katman>1)
   resim=rgb2gray(resim);
end
boyut1=(filtreBoyutu-1)/2;
boyut2=((filtreBoyutu-1)/2)+1;
resim=double(resim);
resim2=double(zeros(row-filtreBoyutu-1,col-filtreBoyutu-1));    
filtre=[0 1 0; 1 -4 1; 0 1 0];
for y=boyut2:row-boyut2
    for x=boyut2:col-boyut2
        komsu=resim((y-boyut1):y+boyut1,x-boyut1:x+boyut1);
        
        toplam=0;
        for m=1:filtreBoyutu
            for n=1:filtreBoyutu
                toplam=toplam+filtre(m,n)*komsu(m,n);
            end
        end
        resim2(y-boyut1,x-boyut1)=toplam;
    end
end

outimg=uint8(resim2);
setappdata(0,'outimg',outimg);
axes(handles.axes2);
imshow(outimg);
else
    msgbox(' Girdiðiniz deger 0 dan büyük 4 den küçük olmalý. Önerilen 3tür');
end


% --- Executes on button press in roberts.
function roberts_Callback(hObject, eventdata, handles)
% hObject    handle to roberts (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
fs = str2double(get(handles.edit1,'String'));
if (fs>0 || fs<0)
   msgbox(' Bu filtrede deðer girmeyin.');
end
resim = getappdata(0,'image');
[~,~,katman]=size(resim);
if(katman>1)
   resim=rgb2gray(resim);
end
res = uint8(resim);  
res = double(res); 
  
resim2 = zeros(size(res)); 

Mx = [1 0; 0 -1]; 
My = [0 1; -1 0]; 
 
for i = 1:size(res, 1) - 1 
    for j = 1:size(res, 2) - 1 

        Gx = sum(sum(Mx.*res(i:i+1, j:j+1))); 
        Gy = sum(sum(My.*res(i:i+1, j:j+1))); 

        resim2(i, j) = sqrt(Gx.^2 + Gy.^2); 
         
    end
end
resim2 = uint8(resim2); 

outimg=uint8(resim2);
setappdata(0,'outimg',outimg);
axes(handles.axes2);
imshow(outimg);

% --- Executes on button press in esikle.
function esikle_Callback(hObject, eventdata, handles)
% hObject    handle to esikle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
esik = str2double(get(handles.edit11,'String'));
if (esik>49 && esik<251)
    resim = getappdata(0,'image');
[str,stn,ktmn]=size(resim);
if (ktmn > 1)
    resim = rgb2gray(resim);
end
outimg = resim;
for i=1:str
    for j=1:stn
        if(resim(i,j) < esik)
            outimg(i,j)=0;
        else
            outimg(i,j)=255;
        end
    end
end
setappdata(0,'outimg',outimg);
axes(handles.axes2);
imshow(outimg);

else
     msgbox(' Girdiðiniz deger 50 den büyük 250 den küçük olmalý');
end

% --- Executes on button press in dondur.
function dondur_Callback(hObject, eventdata, handles)
% hObject    handle to dondur (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

resim = getappdata(0,'image');
[rowsi,colsi,z]= size(resim); 

angle=str2double(get(handles.edit4,'string'));

rads=2*pi*angle/360;  

rowsf=ceil(rowsi*abs(cos(rads))+colsi*abs(sin(rads)));                      
colsf=ceil(rowsi*abs(sin(rads))+colsi*abs(cos(rads)));                     

C=uint8(zeros([rowsf colsf 3 ]));

xo=ceil(rowsi/2);                                                            
yo=ceil(colsi/2);

midx=ceil((size(C,1))/2);
midy=ceil((size(C,2))/2);

for i=1:size(C,1)
    for j=1:size(C,2)                                                       

         x= (i-midx)*cos(rads)+(j-midy)*sin(rads);                                       
         y= -(i-midx)*sin(rads)+(j-midy)*cos(rads);                             
         x=round(x)+xo;
         y=round(y)+yo;

         if (x>=1 && y>=1 && x<=size(resim,1) &&  y<=size(resim,2) ) 
              C(i,j,:)=resim(x,y,:);  
         end

    end
end
 outimg=C;
 setappdata(0,'outimg',outimg);
 axes(handles.axes2);
 imshow(outimg);

% --- Executes on button press in boyut.
function boyut_Callback(hObject, eventdata, handles)
% hObject    handle to boyut (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
x = str2double(get(handles.edit5,'string'));
y = str2double(get(handles.edit6,'string'));
if(x && y)
   resim = getappdata(0,'image');
out = boyutlandir(resim, [x y]);
outimg = out;
setappdata(0,'outimg',outimg);
axes(handles.axes2);
imshow(outimg);
else
    msgbox('piksel deðeri girin');
end


% --- Executes on button press in otele.
function otele_Callback(hObject, eventdata, handles)
% hObject    handle to otele (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

resim = getappdata(0,'image');
shiftX = str2double(get(handles.edit7,'string'));
shiftY = str2double(get(handles.edit8,'string'));
if(shiftX && shiftY)
    nI = uint8( zeros(size(resim,1)+shiftY-1, size(resim,2)+shiftX-1, size(resim,3)));
nI(shiftY:end, shiftX:end, :)  = resim;

outimg= nI;
setappdata(0,'outimg',outimg);
axes(handles.axes2);
imshow(outimg);
else
    msgbox('Kaydýrmak için x ve y degerini girin');
end



% --- Executes on button press in histo.
function histo_Callback(hObject, eventdata, handles)
% hObject    handle to histo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
resim = getappdata(0,'image');
[str,stn,ktmn]=size(resim);
if (ktmn > 1)
    resim = rgb2gray(resim);
end
figure
subplot(2,2,1),imshow(resim),title('Ýlk Resim');
subplot(2,2,2),imhist(resim),title('Resmin Histogrami');
outimg = double(resim);
maxD=max(resim(:));
minD=min(resim(:));
for i=1:str
    for j=1:stn
        nom=double(outimg(i,j)-minD);
        denom=double(maxD-minD);
        outimg(i,j)=(nom*255)/(denom);
    end
end
outimg=uint8(outimg);
subplot(2,2,3),imshow(outimg),title('Çikti Resim');
subplot(2,2,4),imhist(outimg),title('Çikti Resimin Histogrami');
setappdata(0,'outimg',outimg);
axes(handles.axes2);
imshow(outimg);



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in gama.
function gama_Callback(hObject, eventdata, handles)
% hObject    handle to gama (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
for i=0:255;
    f=power((i+0.5)/256,1/2.2);
    LUT(i+1)=uint8(f*256-0.5);
end  
img=getappdata(0,'image');
img0=rgb2ycbcr(img);
R=img(:,:,1);
G=img(:,:,2);
B=img(:,:,3);
Y=img0(:,:,1);
Yu=img0(:,:,1);
[x y]=size(Y);
for row=1:x
    for width=1:y
        for i=0:255
        if (Y(row,width)==i)
             Y(row,width)=LUT(i+1);
             break; 
        end
        end
    end
end
img0(:,:,1)=Y;
img1=ycbcr2rgb(img0);
R1=img1(:,:,1);
G1=img1(:,:,2);
B1=img1(:,:,3);
figure
subplot(3,2,1),imshow(R),title('R Orjinal');
subplot(3,2,2),imshow(R1),title('R1');
subplot(3,2,3),imshow(G),title('G Orjinal');
subplot(3,2,4),imshow(G1),title('G1');
subplot(3,2,5),imshow(B),title('B Orjinal');
subplot(3,2,6),imshow(B1),title('B1');
outimg=img1;
setappdata(0,'outimg',outimg);
axes(handles.axes2);
imshow(outimg);



% --- Executes when selected object is changed in uibuttongroup3.
function uibuttongroup3_SelectionChangedFcn(hObject, eventdata, handles)
% hObject    handle to the selected object in uibuttongroup3 
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
switch get(eventdata.NewValue,'Tag') % Get Tag of selected object.
    case 'radio45'
       set(handles.edit4,'string',45);
     case 'radio90'
       set(handles.edit4,'string',90);
    case 'radio180'
      % a=get(handles.radio180,'string');
       set(handles.edit4,'string',180);
    case 'radio270'
       set(handles.edit4,'string',270);
   
end



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double


% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit7_Callback(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit7 as text
%        str2double(get(hObject,'String')) returns contents of edit7 as a double


% --- Executes during object creation, after setting all properties.
function edit7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit8_Callback(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit8 as text
%        str2double(get(hObject,'String')) returns contents of edit8 as a double


% --- Executes during object creation, after setting all properties.
function edit8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in median.
function median_Callback(hObject, eventdata, handles)
% hObject    handle to median (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
fb = str2double(get(handles.edit1,'String'));
if (fb>0 && fb<4)
if(fb==2)
    fb=3;
end
resim = getappdata(0,'image');
filtreBoyutu=fb;

[row,col,katman]=size(resim);
if(katman>1)
    resim=rgb2gray(resim);
end

genisleme=(filtreBoyutu-1)/2+1;
baslama=((filtreBoyutu-1)/2);

resim2=zeros(row+genisleme,col+genisleme);  
[row1,col1]=size(resim2);

for i=2:row1-genisleme
    for j=2:col1-genisleme
        resim2(i,j)=resim(i,j);
    end
end
for y=2:row1-baslama
  for x=2:col1-baslama
        komsu=resim2((y-baslama):y+baslama,(x-baslama):x+baslama);
        siraliVektor=sort(komsu(:));
        yeniDeger=median(siraliVektor);
        resim2(y,x)=yeniDeger;
  end
end
outimg=uint8(resim2);
setappdata(0,'outimg',outimg);
axes(handles.axes2);
imshow(uint8(resim2));
else
    msgbox(' Girdiðiniz deger 0 dan büyük 4 den küçük olmalý');
end



function edit11_Callback(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit11 as text
%        str2double(get(hObject,'String')) returns contents of edit11 as a double


% --- Executes during object creation, after setting all properties.
function edit11_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
