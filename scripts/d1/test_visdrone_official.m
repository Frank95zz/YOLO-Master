function result = test_visdrone_official(toolkit, dataset, predictions, outputDir, toolkitManifest)
% Compare the wrapper with the official entry's exact saveAnnoRes/calcAccuracy path.
if ~exist(outputDir, 'dir'), mkdir(outputDir); end
report = evaluate_visdrone_official(toolkit, dataset, predictions, fullfile(outputDir, 'real-wrapper.json'), toolkitManifest);
addpath(fullfile(toolkit,'utils'));
if exist('mean2','file') == 0
    addpath(fullfile(fileparts(mfilename('fullpath')),'visdrone_matlab_compat'));
end
names = findImageList(fullfile(dataset,'annotations'));
[gt,dt] = saveAnnoRes(fullfile(dataset,'annotations'), predictions, fullfile(dataset,'images'), numel(names), names);
[a,b,c,d,e,f,g] = calcAccuracy(numel(names),gt,dt);
reference = [a,b,c,d,e,f,g] / 100;
actual = [report.metrics.AP_all,report.metrics.AP_50,report.metrics.AP_75,report.metrics.AR_1,...
          report.metrics.AR_10,report.metrics.AR_100,report.metrics.AR_500];
error = max(abs(reference-actual));
assert(error <= 1e-6, 'Wrapper differs from official MATLAB reference');

% Exact IoU threshold, duplicate detection and score=0 ignored target semantics.
[~,detection] = evalRes([10,10,20,20,0], [10,10,10,20,.9;10,10,20,20,.8], .5);
assert(isequal(detection(:,6),[1;0]), 'IoU boundary or duplicate match differs');
[~,detection] = evalRes([10,10,20,20,1], [12,12,5,5,.9], .5);
assert(detection(1,6) == -1, 'Ignored GT match differs');
original = [1,1,30,30,0,0,0,0;50,50,10,10,1,1,0,0;60,60,10,10,0,11,0,0];
[g,d] = dropObjectsInIgr(original,[5,5,10,10,.9,1,-1,-1;50,50,10,10,.8,1,-1,-1],100,100);
assert(size(d,1)==1 && d(1,1)==50 && ~any(g(:,6)==0), 'Ignore-region filtering differs');

emptyDataset = fullfile(outputDir,'empty-fixture');
emptyPred = fullfile(outputDir,'empty-predictions');
mkdir(fullfile(emptyDataset,'annotations')); mkdir(fullfile(emptyDataset,'images')); mkdir(emptyPred);
rows = zeros(10,8);
for k=1:10, rows(k,:)=[k*5,5,3,3,1,k,0,0]; end
writematrix(rows,fullfile(emptyDataset,'annotations','synthetic.txt'));
imwrite(zeros(64,64,3,'uint8'),fullfile(emptyDataset,'images','synthetic.jpg'));
fid=fopen(fullfile(emptyPred,'synthetic.txt'),'w'); fclose(fid);
fid=fopen(fullfile(emptyPred,'export.json'),'w'); fprintf(fid,'{}'); fclose(fid);
emptyReport = evaluate_visdrone_official(toolkit,emptyDataset,emptyPred,fullfile(outputDir,'empty-wrapper.json'),toolkitManifest);
assert(all(structfun(@(x) x==0,emptyReport.metrics)), 'Empty predictions must score zero');
result.status='PASSED';
result.toolkit_commit=report.toolkit_commit;
result.real_image_count=report.image_count;
result.official_vs_wrapper_max_abs_0_1=error;
result.tolerance_0_1=1e-6;
result.tests={'fixed_real_images_vs_official_entry','exact_iou_threshold','duplicate_prediction',...
              'score0_ignore_match','ignore_region_filter','empty_predictions_10_classes'};
result.fixture_not_model_accuracy=true;
result.matlab_version=version;
fid=fopen(fullfile(outputDir,'evaluator-tests.json'),'w');
guard=onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(result,PrettyPrint=true));
disp(result);
end
