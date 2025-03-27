function varargout = BrainTumor_Detection(varargin)
% BRAINTUMOR_DETECTION MATLAB code for BrainTumor_Detection.fig

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name', mfilename, ...
                   'gui_Singleton', gui_Singleton, ...
                   'gui_OpeningFcn', @BrainTumor_Detection_OpeningFcn, ...
                   'gui_OutputFcn', @BrainTumor_Detection_OutputFcn, ...
                   'gui_LayoutFcn', [] , ...
                   'gui_Callback', []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

%% Opening Function
function BrainTumor_Detection_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;

% Initialize handles for image processing
handles.imagePaths = {};
handles.currentImageIndex = 0;
handles.results = {};

% Update handles structure
guidata(hObject, handles);

%% Output Function
function varargout = BrainTumor_Detection_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

%% Upload Callback
function upload_Callback(hObject, eventdata, handles)
folder_name = uigetdir;
if folder_name
    image_files = dir(fullfile(folder_name, '*.jpg'));
    [~, sortedIdx] = sort({image_files.name});
    image_files = image_files(sortedIdx);
    handles.imagePaths = fullfile(folder_name, {image_files.name});
    handles.currentImageIndex = 1;
    handles.results = processImages(handles.imagePaths);
    guidata(hObject, handles);
    displayResults(handles);
end

%% Previous Button Callback
function previous_Callback(hObject, eventdata, handles)
if handles.currentImageIndex > 1
    handles.currentImageIndex = handles.currentImageIndex - 1;
    guidata(hObject, handles);
    displayResults(handles);
end

%% Next Button Callback
function next_Callback(hObject, eventdata, handles)
if handles.currentImageIndex < length(handles.results)
    handles.currentImageIndex = handles.currentImageIndex + 1;
    guidata(hObject, handles);
    displayResults(handles);
end

%% Image Processing Function
function results = processImages(imagePaths)
numImages = length(imagePaths);
results = cell(1, numImages);  % Initialize the results cell array
for i = 1:numImages
    imgPath = imagePaths{i};
    img = imread(imgPath);
    
    % --- Step 1: Convert to grayscale if necessary
    if size(img, 3) > 1
        inp = rgb2gray(img);
    else
        inp = img;
    end
    
    % --- Step 2: Anisotropic Diffusion
    num_iter = 10; delta_t = 1/7; kappa = 15; option = 2;
    inp = anisodiff_function(inp, num_iter, delta_t, kappa, option);  % Call the function from the separate file
    inp = uint8(inp);
    inp = imresize(inp, [256, 256]);
    
    % --- Step 3: K-means Clustering
    preprocessedImg = imgaussfilt(inp, 2);
    num_clusters = 3;
    [clusteredImg, clusterCenters] = kmeans(double(preprocessedImg(:)), num_clusters);
    clusteredImg = reshape(clusteredImg, size(preprocessedImg));
    segmentedImg = zeros(size(preprocessedImg));
    for k = 1:num_clusters
        segmentedImg(clusteredImg == k) = clusterCenters(k);
    end
    
    % --- Step 4: Region Growing
    seedPoint = [100, 100];
    regionImg = grayconnected(inp, seedPoint(1), seedPoint(2));
    
    % --- Step 5: Edge Detection
    edgesImg = edge(inp, 'sobel');
    
    % --- Step 6: Thresholding and Tumor Detection
    sout = imresize(inp, [256, 256]);
    t0 = mean(inp(:));
    th = t0 + ((max(inp(:)) + min(inp(:))) / 2);
    for j = 1:numel(inp)
        if inp(j) > th
            sout(j) = 1;
        else
            sout(j) = 0;
        end
    end
    label = bwlabel(sout);
    stats = regionprops(logical(sout), 'Solidity', 'Area', 'BoundingBox');
    density = [stats.Solidity];
    area = [stats.Area];
    high_dense_area = density > 0.7;
    max_area = max(area(high_dense_area));
    tumor_label = find(area == max_area);
    tumor = ismember(label, tumor_label);
    
    if max_area > 200
        tumorAlone = tumor;
        wantedBox = stats(tumor_label).BoundingBox;
    else
        tumorAlone = zeros(size(tumor));
        wantedBox = [];
    end
    
    % --- Step 7: Tumor Outline
    if max_area > 200
        dilationAmount = 5;
        rad = floor(dilationAmount);
        [r, c] = size(tumorAlone);
        filledImage = imfill(tumorAlone, 'holes');
        erodedImage = imerode(filledImage, strel('disk', rad));
        tumorOutline = tumorAlone;
        tumorOutline(erodedImage > 0) = 0;
        rgb = cat(3, inp, inp, inp);
        red = rgb(:, :, 1); red(tumorOutline) = 255;
        green = rgb(:, :, 2); green(tumorOutline) = 0;
        blue = rgb(:, :, 3); blue(tumorOutline) = 0;
        tumorOutlineInserted = cat(3, red, green, blue);
    else
        tumorOutline = zeros(size(inp));
        tumorOutlineInserted = img;  % Show original image with no tumor detected caption
    end
    
    % Save the results
    results{i} = struct('img', img, 'inp', inp, 'segmentedImg', segmentedImg, ...
        'regionImg', regionImg, 'edgesImg', edgesImg, 'sout', sout, ...
        'tumorAlone', tumorAlone, 'wantedBox', wantedBox, 'tumorOutline', tumorOutline, ...
        'tumorOutlineInserted', tumorOutlineInserted);
end

%% Display Results Function
function displayResults(handles)
if handles.currentImageIndex > 0 && handles.currentImageIndex <= length(handles.results)
    result = handles.results{handles.currentImageIndex};
    axes(handles.axes1); imshow(result.img); title('Input Image', 'FontSize', 20);
    axes(handles.axes2); imshow(result.inp); title('Filtered Image', 'FontSize', 20);
    axes(handles.axes3); imshow(result.segmentedImg, []); title('K-means Clustering', 'FontSize', 20);
    axes(handles.axes4); imshow(result.regionImg); title('Region Growing', 'FontSize', 20);
    axes(handles.axes5); imshow(result.edgesImg); title('Edge Detection', 'FontSize', 20);
    
    % New axes for Threshold Techniques
    axes(handles.axes6); imshow(result.sout, []); title('Global Threshold', 'FontSize', 20);
    
    if isempty(result.wantedBox)
        axes(handles.axes7); imshow(result.img); title('No Tumor Detected', 'FontSize', 20);
    else
        axes(handles.axes7); imshow(result.inp); title('Tumor Area', 'FontSize', 20);
        hold on; rectangle('Position', result.wantedBox, 'EdgeColor', 'y'); hold off;
    end
    axes(handles.axes8); imshow(result.tumorAlone); title('Isolated Tumor', 'FontSize', 20);
    axes(handles.axes9); imshow(result.tumorOutline); title('Tumor Outline', 'FontSize', 20);
    if isfield(result, 'tumorOutlineInserted')
        axes(handles.axes10); imshow(result.tumorOutlineInserted); 
        if isempty(result.wantedBox)
            title('No Tumor Detected', 'FontSize', 20);
        else
            title('Detected Tumor', 'FontSize', 20);
        end
    else
        axes(handles.axes10); imshow(result.img); title('No Tumor Detected', 'FontSize', 20);
    end
end
