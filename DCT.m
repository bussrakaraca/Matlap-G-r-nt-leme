function [resim2] = DCT( resim)
    resim = double(resim)/255;
    [~,~,katman]=size(resim);
    if(katman>1)
        resim=GriSeviyeDonusum(resim);
    end
    
    resdct=dct2(resim);
    resus=(resdct).^2;
    resus=resus(:);
    [~,index]=sort(resus);
    index=flipud(index);
    sikistirma=zeros(size(resim));
    ka = 2000;
    for k=1:ka
        sikistirma(index(k))=resdct(index(k));
    end
    im=idct2(sikistirma);
    resim2=im;
    imwrite(resim2,'dct.jpg');
end