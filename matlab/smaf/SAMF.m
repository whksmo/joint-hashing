classdef SAMF < handle
    properties (SetAccess = private)
       %  padding = [1.5 1.5 1.5 1.5 1.5 2.5 3.5  4.5 5.5 6.5];  %extra area surrounding the target
        padding = [1.5 1.5 1.5 1.5 1.5 2 2.5 3.5 4.5 5 5.5 6];
        lambda = 1e-4;  %regularization
        output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)        
        interp_factor = 0.02;
        kernel_sigma = 0.1; %gaussian kernel bandwidth        
        hog_orientations = 9;
        hog_cell_size = 4;
        
        search_size = [1  0.985 0.99 0.995 1.005 1.01 1.015];
        szid = 1;
        paddingid=1;
        center;
        target_sz;
        init_sz;
        box;
        
        window_sz;
        output_sigma;
        yf;
        cos_window;
        response;
        occ_num=0;
        
        model_alphaf;
        model_xf;
        model_patch;
        current_im;
    end
    
    methods
        function tracker = SAMF(im, center, target_sz)
            tracker.current_im = im;
            
            tracker.center = center; % center: (row,col),i.e.,(y,x)
            tracker.target_sz = target_sz; % target_sz: (rows, cols), i.e., (height, width)
            tracker.init_sz=target_sz;
            tracker.box = [ center([2 1]) - target_sz([2 1])/2, target_sz([2 1]) ];%(x,y,width,height)
            for i=1:size(tracker.padding,2)
                tracker.window_sz{i} = target_sz * (1 + tracker.padding(i));
                tracker.output_sigma = sqrt(prod(target_sz)) * tracker.output_sigma_factor / tracker.hog_cell_size;
                tracker.yf{i} = fft2(gaussian_shaped_labels(tracker.output_sigma, floor(tracker.window_sz{i} / tracker.hog_cell_size)));
                tracker.cos_window{i} = hann(size(tracker.yf{i},1)) * hann(size(tracker.yf{i},2))';

                tracker.response{i} = zeros(size(tracker.cos_window{i},1),size(tracker.cos_window{i},2),size(tracker.search_size,2));

                target_sz = target_sz * tracker.search_size(tracker.szid);
                tmp_sz = (target_sz * (1 + tracker.padding(i)));
                patch = mexResize( get_subwindow(im,center,tmp_sz), tracker.window_sz{i}, 'auto');
                xf = fft2(get_features(patch, tracker.cos_window{i}));% appearance model
                kf = gaussian_correlation(xf, xf, tracker.kernel_sigma);
                alphaf = tracker.yf{i} ./ (kf + tracker.lambda);   %equation for fast training
                tracker.model_xf{i} = xf;
                tracker.model_alphaf{i} = alphaf;
                tracker.model_patch{i} = patch;
                
            end
        end
        
        function [box,response,mat] = track(self, im)
            self.current_im = im;
            global resp;
            
           if resp<0.2
              
               self.paddingid=self.paddingid+1;
               self.paddingid(self.paddingid>10)=10;
               self.occ_num=self.occ_num+1;
               
            else
                self.paddingid=self.paddingid-2;
                self.paddingid(self.paddingid<1)=1;
            end
%             if resp<0.2
%                 if resp>0.1
%                     self.paddingid=self.paddingid+1;
%                     self.paddingid(self.paddingid>6)=6;
%                     self.occ_num=self.occ_num+1;
%                 else
%                     self.paddingid=self.paddingid+2;
%                     self.paddingid(self.paddingid>1)=1;
%                     self.occ_num=self.occ_num+1;
%                 end  
%             else
%                 self.paddingid=self.paddingid-2;
%                 self.paddingid(self.paddingid<1)=1;
%             end
%                 self.window_sz=self.init_sz*(1+self.padding(self.paddingid));
%             self.interp_factor=1./(1+exp(16-20*resp));
%             self.interp_factor=resp^6;
%             if self.occ_num<10
                for i=1:size(self.search_size,2)
                    tmp_sz = (self.target_sz * (1 + self.padding(self.paddingid))) * self.search_size(i);
                    patch = mexResize( get_subwindow(im,self.center,tmp_sz), self.window_sz{self.paddingid}, 'auto');
                    zf = fft2(get_features(patch, self.cos_window{self.paddingid}));
                    kzf = gaussian_correlation(zf, self.model_xf{self.paddingid}, self.kernel_sigma);
                    self.response{self.paddingid}(:,:,i) = real(ifft2(self.model_alphaf{self.paddingid} .* kzf));  %equation for fast detection
                end
                response=max(self.response{self.paddingid}(:));
                [vert_delta,tmp, id] = find(self.response{self.paddingid} == max(self.response{self.paddingid}(:)), 1);
                mat=fftshift(self.response{self.paddingid}(:,:,id));
                self.szid = floor((tmp-1)/(size(self.cos_window{self.paddingid},2)))+1;

                horiz_delta = tmp - ((self.szid -1)* size(self.cos_window{self.paddingid},2));
                if vert_delta > size(zf,1) / 2  %wrap around to negative half-space of vertical axis
                    vert_delta = vert_delta - size(zf,1);
                end
                if horiz_delta > size(zf,2) / 2  %same for horizontal axis
                    horiz_delta = horiz_delta - size(zf,2);
                end

                tmp_sz = floor((self.target_sz * (1 + self.padding(self.paddingid)))*self.search_size(self.szid));
                current_size = tmp_sz(2)/self.window_sz{self.paddingid}(2);
                self.center = self.center + current_size * self.hog_cell_size * [vert_delta - 1, horiz_delta - 1];
                self.target_sz = self.target_sz * self.search_size(self.szid);
                position = self.center([2 1]) - self.target_sz([2 1])/2;
                position(position<0)=0;
                self.center([2 1])=position+self.target_sz([2 1])/2;
                box = [position, self.target_sz([2 1])];
                self.box = box; %(x,y,width,height)
%             else
%                 self.box=[size(im,1)/2-self.target_sz(1)/2,size(im,2)/2-self.target_sz(2)/2,self.target_sz([2 1])];
%                 response=0;
%                 mat=0;
%             end
                
        end
        
        function update(self)           
            tmp_sz = (self.target_sz * (1 + self.padding(self.paddingid)));
            patch = mexResize( get_subwindow(self.current_im,self.center,tmp_sz), self.window_sz{self.paddingid}, 'auto');
            xf = fft2(get_features(patch, self.cos_window{self.paddingid}));% appearance model
            kf = gaussian_correlation(xf, xf, self.kernel_sigma);
            alphaf = self.yf{self.paddingid} ./ (kf + self.lambda);   %equation for fast training
            
            self.model_alphaf{self.paddingid} = (1 - self.interp_factor) * self.model_alphaf{self.paddingid} + self.interp_factor * alphaf;
            self.model_xf{self.paddingid} = (1 - self.interp_factor) * self.model_xf{self.paddingid} + self.interp_factor * xf;
            
         
            self.model_patch{self.paddingid} = (1 - self.interp_factor) * self.model_patch{self.paddingid} + self.interp_factor * patch;
            global frame2
            global name;
            
      %       imwrite(self.model_patch{self.paddingid},['.\template\SAMF_OD\',name,'\','template',int2str(frame2),'.jpg']);

            
        end
    end
end
