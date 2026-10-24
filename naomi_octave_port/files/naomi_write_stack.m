function naomi_write_stack(movie,path,offset,scale)
% Write all pages together. GraphicsMagick append can promote later gray16
% pages to RGB16, which many readers expose as RGB8.
encoded=uint16(round((double(movie)-offset)*scale));
encoded=reshape(encoded,size(movie,1),size(movie,2),1,size(movie,3));
imwrite(encoded,path,'tif','Compression','none');
end
