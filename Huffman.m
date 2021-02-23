function [res2] = Huffman(res)
[m,n,katman]=size(res);
    if(katman>1)
        res=GriSeviyeDonusum(res);
    end
I=res;
Totalcount=m*n;

cnt=1;
sigma=0;

for i=0:255
k=I==i;
count(cnt)=sum(k(:));

pro(cnt)=count(cnt)/Totalcount;
sigma=sigma+pro(cnt);
cumpro(cnt)=sigma;
cnt=cnt+1;
end

symbols =[0:255];

dict = huffmandict(symbols,pro);

vec_size = 1;
for p = 1:m
for q = 1:n
newvec(vec_size) = I(p,q);
vec_size = vec_size+1;
end
end
hcode = huffmanenco(newvec,dict);
dhsig1 = huffmandeco(hcode,dict);
dhsig = uint8(dhsig1);
dec_row=sqrt(length(dhsig));
dec_col=dec_row;

arr_row = 1;
arr_col = 1;
vec_si = 1;
for x = 1:m
for y = 1:n
back(x,y)=dhsig(vec_si);
arr_col = arr_col+1;
vec_si = vec_si + 1;
end
arr_row = arr_row+1;
end

[deco, map] = gray2ind(back,256);
RGB = ind2rgb(deco,map);
imwrite(RGB,'huffman.jpg');
res2=RGB;
end