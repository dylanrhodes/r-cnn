function extract_region_feats()

splits = {'train', 'test'};

CROP_SIZE = 227;
MARGIN = 16;
BATCH_SIZE = 100;
CODE_LENGTH = 512;

init_key = caffe('init', 'cnn_deploy.prototxt', 'cnn512.caffemodel', 'test');
caffe('set_device', 4);
caffe('set_mode_gpu');

for s = 1:numel(splits)
  split = splits{s};
  im_data = load(sprintf('%s_ims.mat', split));
  images = im_data.images;

  ssearch_data = load(sprintf('ssearch_%s.mat', split));
  ssearch_boxes = ssearch_data.ssearch_boxes;
  
  % Get rid of test bounding boxes, just so we're not tempted to use them
  if strcmp(split, 'test')
    [images.bboxes] = deal([]);
    [images.classes] = deal([]);
  end

  mean_data = load('ilsvrc_2012_mean.mat');
  image_mean = mean_data.image_mean;
  off = floor((size(image_mean, 1) - CROP_SIZE) / 2) + 1;
  image_mean = image_mean(off: off + CROP_SIZE - 1, off: off + CROP_SIZE - 1, :);
  
  for i = 1:numel(images)
      out_file = ['../features/' images(i).fname(1:size(images(i).fname, 2) - 4) '.bin'];
      
      if exist(out_file, 'file')
          fprintf('Skipping %s...\n', out_file);
          continue
      else
          fprintf('Collecting features for %s...\n', out_file);
          tic
      end
      
      im = imread(['../images/' images(i).fname]);
      im = single(im(:, :, [3 2 1]));
      
      all_box = [images(i).bboxes; ssearch_boxes{i}];
      
      % Dilate bounding boxes
      width = all_box(:, 3) - all_box(:, 1);
      height = all_box(:, 4) - all_box(:, 2);
      width_pad = ceil(width * (MARGIN / CROP_SIZE / 2));
      height_pad = ceil(height * (MARGIN / CROP_SIZE / 2));
      
      all_box(:, 1) = all_box(:, 1) - width_pad;
      all_box(:, 3) = all_box(:, 3) + width_pad;
      all_box(:, 2) = all_box(:, 2) - height_pad;
      all_box(:, 4) = all_box(:, 4) + height_pad;
      
      % Pad image with large negative number
      max_pad = max([max(width_pad) max(height_pad)]);
      im = padarray(im, [max_pad max_pad 0], -1000);
      all_box = all_box + max_pad;
      
      all_codes = zeros(size(all_box, 1), CODE_LENGTH);
      
      for j = 1:ceil(size(all_box, 1) / BATCH_SIZE)
          s_idx = (j-1) * BATCH_SIZE + 1;
          e_idx = min(j * BATCH_SIZE, size(all_box, 1));
          batch_len = e_idx - s_idx + 1;
          
          boxes = floor(all_box(s_idx:e_idx, :));      
          regions = zeros(CROP_SIZE, CROP_SIZE, 3, BATCH_SIZE, 'single');
          
          for k = 1:batch_len
              patch = im(boxes(k,2):boxes(k,4), boxes(k,1):boxes(k,3), :);
              patch = imresize(patch, [CROP_SIZE CROP_SIZE], 'bilinear', 'antialiasing', false);
              patch = patch - image_mean;
              patch(patch < -256) = 0.0;
              regions(:, :, :, k) = permute(single(patch), [2 1 3]);
          end
      
          f = caffe('forward', {regions});
          feat = single(reshape(f{1}(:), [], size(regions, 4)));
          all_codes(s_idx:e_idx, :) = feat(:,1:batch_len)';
      end
      
      fwrite(out_file, all_codes, 'single');
      fprintf('Wrote data file to %s...\n', out_file);
      toc
  end
end
