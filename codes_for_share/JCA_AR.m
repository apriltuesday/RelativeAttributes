function [classification_accuracy] = JCA_AR(weight_flag,active_version,uniform_flag,weight_active_flag)

%%% In this file, we write a code to learn the category models and
%%% attribute models simultaneously for a dataset. We assume that none of
%%% the data points are labeled and we do not have any idea about the
%%% attributes. We start with picking up a random image , and show to some
%%% user say that we believe this image is from a certain class c. 
%%% But since we did not have have any training image, it will random. If
%%% the answer is right, user says yes (initially the chance of which is very 
%%% low), if the answer is wrong, user says no, gives the correct label and
%%% also gives a reason for that the image classification being wrong.

%%% One important issue in this case is to keep track of the fact that any
%%% conclusion drawn from attribute based user feedback needs to be updated
%%% as the attribute model gets updated. 

%%% We also assume that user knows labels of each image
 
 close all; clc;
 
 tic
 
 load data.mat; 
  
 load train_test_and_validation_ids_multiple.mat; % Loading the initial set of training labels
 
 load human_attribute_results_1.mat; % Loading this will replace the earlier attribute_names

 no_attribute=length(attribute_names); no_images=length(im_names);
 
 feat_dim=size(feat,2);

 clearvars relative_att_predictions relative_att_predictor;
 
 NRUN=1; % We run multiple times and see the average accuracy increment
 
 total_increment_in_training_size=300;
 
 classification_accuracy=zeros(NRUN,total_increment_in_training_size);
 
 for nrun=1:1:NRUN
 
 %%%%%%%%%%%%%%%% We store the answers here
 
 max_no_questions=no_train_images;
 
 correctness=zeros(max_no_questions,1); % If correct answer index will be 1 and if 
 
 % wrong, index will be zero
 
 % User will derive information from the category level attribute
 % comparisons and will feed the system, but this includes all information
 % which can be derived from what the user said
 
 attribute_based_feedback=zeros(10000,3); % The first elment has
 
 % some attribute 'a' more than the second element and the third element
 % stores the name of the attribute
 
 no_attr_based_feedback=0; 
 
 % This is an answer tracker and stores every user answer
 
 answer_tracker=zeros(max_no_questions,5);
 
 attribute_based_feedback_2=zeros(no_class,no_images,2); % Storage for pairwise relation derivation
 
 threshold_qsns=2; % This is the number of questions after which
 
 % we start learning a classifier.
 
 %%%%%%%%%%%%%%%%
 
 %%
 
 %%% We define the preference matrices here and 
 
 % Update the attribute model based on the initial training data
 
 %%%%%%%%%%%%%%%%
 
 %%
 
 %%% We run an iteration on the number of training images and gradually
 %%% increase it to see how the classification accuracy increase
     
relative_attribute_predictor=zeros(size(feat,2),length(attribute_names));
 
relative_attribute_predictions=feat*relative_attribute_predictor;
     
     %%
     
 variable_train_ids=multiple_initial_train_ids(nrun,:);
 
 train_ids=multiple_train_ids(nrun,:);
 
 test_ids=multiple_test_ids(nrun,:);
 
 permanent_train_ids=train_ids; 
 
 unlabeled_ids=setdiff(permanent_train_ids,variable_train_ids);
    
 rand_index=randperm(length(unlabeled_ids));
 
 unlabeled_ids=unlabeled_ids(rand_index);  
  
 answer_tracker_2=ones(length(unlabeled_ids),no_class);
  
 %%% In the following matrix we store which unlabeled examples can be
 %%% considered as negative examples and for which class, once we get a
 %%% label for an image we delete that corresponding row from the following
 %%% matrix. Example which can be considered as negative are marked as +1
 %%% and others are kept zero
 
 negative_examples=zeros(length(permanent_train_ids)-length(variable_train_ids),no_class);
 
  negative_examples_weights=...
     zeros(length(permanent_train_ids)-length(variable_train_ids),no_class);
 
   negative_examples_weights_normalized=...
     zeros(length(permanent_train_ids)-length(variable_train_ids),no_class);
 
 predicted_probabilities=zeros(length(test_ids)+length(unlabeled_ids),no_class);
 
 store_attribute_accuracy_on_the_fly=zeros(300,1);
 
 store_attribute_accuracy_pretrained=zeros(300,1);
 
 for ts=1:1:total_increment_in_training_size

 %%% Defining unlabeled ids
 
 %unlabeled_ids=setdiff(permanent_train_ids,variable_train_ids);

  %%% Finding the best possible image id
  
  display('waiting for the best image to ask')
  
  if active_version==-1
      
       image_id=unlabeled_ids(1); image_id_in_unlabeled_ids=1;
      
  elseif active_version==0    
        
      [image_id, image_id_in_unlabeled_ids] =...
    most_informative_image_id_version_0(unlabeled_ids, variable_train_ids,...
    predicted_probabilities(length(test_ids)+1:length(test_ids)+...
        length(unlabeled_ids),:),no_class);
      
  elseif active_version==1
  
  if weight_active_flag==0

% Only labeled are considered

   [image_id, image_id_in_unlabeled_ids] =...
    most_informative_image_id_new_1(unlabeled_ids, variable_train_ids,...
    predicted_probabilities(length(test_ids)+1:length(test_ids)+...
    length(unlabeled_ids),:),relative_attribute_predictions,no_class,ts,uniform_flag);

  elseif weight_active_flag==1
      
    [image_id, image_id_in_unlabeled_ids] =...
    most_informative_image_id_new_1_softer(unlabeled_ids, variable_train_ids,...
    predicted_probabilities(length(test_ids)+1:length(test_ids)+...
    length(unlabeled_ids),:),relative_attribute_predictions,no_class,negative_examples_weights,ts);
      
  end
  
  elseif active_version==2

% Both labels and attributes are considered
      
      [image_id, image_id_in_unlabeled_ids] =...
    most_informative_image_id_joint_new_fast_6(unlabeled_ids, variable_train_ids,...
    predicted_probabilities(length(test_ids)+1:length(test_ids)+...
    length(unlabeled_ids),:),relative_attribute_predictions,relative_attribute_predictor,no_class,...
    attribute_based_feedback,no_attr_based_feedback,uniform_flag);

   elseif active_version==5
      
      [image_id, image_id_in_unlabeled_ids] =...
    most_informative_image_id_partition_new_1(unlabeled_ids, variable_train_ids,...
    predicted_probabilities(length(test_ids)+1:length(test_ids)+...
    length(unlabeled_ids),:),relative_attribute_predictions,relative_attribute_predictor,no_class,...
    attribute_based_feedback,no_attr_based_feedback,ts,uniform_flag,nrun);

    elseif active_version==6
      
      [image_id, image_id_in_unlabeled_ids] =...
    most_informative_image_id_joint_new_2(unlabeled_ids, variable_train_ids,...
    predicted_probabilities(length(test_ids)+1:length(test_ids)+...
    length(unlabeled_ids),:),relative_attribute_predictions,relative_attribute_predictor,no_class,...
    attribute_based_feedback,no_attr_based_feedback,ts);

  end

%keyboard
 
  %%% Getting feedback from the user
  
  useful_id=image_id_in_unlabeled_ids;
  
  if size(unique(class_labels(variable_train_ids)),1)<2 || ts ==1
      
      predicted_label_random_image=randi(length(class_names));
      
  else
      
      predicted_label_random_image=predicted_labels(length(test_ids)+useful_id);
      
  end
  
  %%%
  
  [correct_index, attribute_index,much_or_less_index] =...
human_oracle_based_on_attribute(image_id, predicted_label_random_image);

answer_tracker(ts,:)=[image_id predicted_label_random_image correct_index...
    attribute_index much_or_less_index]; 

if correct_index==0
    
    attribute_based_feedback_2(predicted_label_random_image,image_id,1)=much_or_less_index;
    
    attribute_based_feedback_2(predicted_label_random_image,image_id,2)=attribute_index;
    
end

%keyboard

%%% In this section we incorporate the negative example decision derived
%%% from a user's attribute based feedback

%%

if correct_index==0

% NOW WE ONLY INCLUDE CONSTRAINTS BASED ON THE SAMPLE WE ASKED TO THE USER
% AND THE TRAINING SAMPLES FROM ITS PREDICTED CLASS

%% We can constrain the maximum number of pairs, after which we stop updating the attribute model

max_no_pairs_for_attribute_model=5000;

vtd=1:1:length(variable_train_ids);

vtd_predicted_class=...
    vtd(class_labels(variable_train_ids)==predicted_label_random_image);

% Update 1

for v=1:1:length(vtd_predicted_class)
    
        no_attr_based_feedback=no_attr_based_feedback+1;
        
        attribute_based_feedback(no_attr_based_feedback,3)=attribute_index;
    
if much_or_less_index==+1
    
        attribute_based_feedback(no_attr_based_feedback,1)=variable_train_ids(vtd_predicted_class(v));
        
        attribute_based_feedback(no_attr_based_feedback,2)=image_id;
    
else

        attribute_based_feedback(no_attr_based_feedback,2)=variable_train_ids(vtd_predicted_class(v));
        
        attribute_based_feedback(no_attr_based_feedback,1)=image_id;

end

end

% Update 2

temp_index=1:1:no_images;

relations_involved_with_image_id=temp_index(attribute_based_feedback_2(class_labels(image_id),:,1)~=0);

for riwii=1:1:length(relations_involved_with_image_id)
    
   no_attr_based_feedback=no_attr_based_feedback+1;
   
   attribute_based_feedback(no_attr_based_feedback,3)=...
       attribute_based_feedback_2(class_labels(image_id),relations_involved_with_image_id(riwii),2);
   
   if attribute_based_feedback_2(class_labels(image_id),relations_involved_with_image_id(riwii),1)==+1
       
       attribute_based_feedback(no_attr_based_feedback,1)=image_id;
       
       attribute_based_feedback(no_attr_based_feedback,2)=relations_involved_with_image_id(riwii);
       
   elseif attribute_based_feedback_2(class_labels(image_id),relations_involved_with_image_id(riwii),1)==-1
       
       attribute_based_feedback(no_attr_based_feedback,2)=image_id;
       
       attribute_based_feedback(no_attr_based_feedback,1)=relations_involved_with_image_id(riwii);
       
   end
    
end

if no_attr_based_feedback>max_no_pairs_for_attribute_model
    
    rand_temp=randperm(no_attr_based_feedback);
    
    attribute_based_feedback(1:no_attr_based_feedback,:)=attribute_based_feedback(rand_temp,:);
    
     [relative_attribute_predictor, relative_attribute_predictions]=...
     learn_the_attribute_model(attribute_based_feedback,max_no_pairs_for_attribute_model);
    
else

 [relative_attribute_predictor, relative_attribute_predictions]=...
     learn_the_attribute_model(attribute_based_feedback,no_attr_based_feedback);
 
end 
 
 
%%

negative_examples=zeros(length(permanent_train_ids)-length(variable_train_ids),no_class);

negative_examples_weights=...
     zeros(length(permanent_train_ids)-length(variable_train_ids),no_class);

relative_attributes_unlabeled_data=relative_attribute_predictions(unlabeled_ids,:);

uid=1:1:length(unlabeled_ids);

for no_ans=1:1:ts
    
    if answer_tracker(no_ans,3)==0 % If the answer is wrong only
    
     unlabeled_attribute_strength=...
         relative_attributes_unlabeled_data(:,answer_tracker(no_ans,4));
     
     img_id_attribute_strength=relative_attribute_predictions(answer_tracker(no_ans,1),...
         answer_tracker(no_ans,4));
     
     if answer_tracker(no_ans,5)==+1
         
     temp=uid(unlabeled_ids(:)==answer_tracker(no_ans,1));
         
     if isempty(temp)
     
     ids_more_attribute= uid(unlabeled_attribute_strength(:)>img_id_attribute_strength);
     
     else
         
     ids_more_attribute= [uid(unlabeled_attribute_strength(:)>img_id_attribute_strength) temp];
         
     end
     
     [~,sorted_index]=sort(unlabeled_attribute_strength(ids_more_attribute));
     
     negative_examples(ids_more_attribute,answer_tracker(no_ans,2))=1;
     
     negative_examples_weights(ids_more_attribute(sorted_index),answer_tracker(no_ans,2))=...
         negative_examples_weights(ids_more_attribute(sorted_index),answer_tracker(no_ans,2))+[1:1:size(ids_more_attribute,2)]';
     
     else
         
     temp=uid(unlabeled_ids(:)==answer_tracker(no_ans,1));
         
     if isempty(temp)
     
     ids_less_attribute= uid(unlabeled_attribute_strength(:)<img_id_attribute_strength);
     
     else
         
     ids_less_attribute= [uid(unlabeled_attribute_strength(:)<img_id_attribute_strength) temp];
         
     end
     
     [~,sorted_index]=sort(unlabeled_attribute_strength(ids_less_attribute),'descend');
     
     negative_examples(ids_less_attribute,answer_tracker(no_ans,2))=1;
     
     negative_examples_weights(ids_less_attribute(sorted_index),answer_tracker(no_ans,2))=...
         negative_examples_weights(ids_less_attribute(sorted_index),answer_tracker(no_ans,2))+[1:1:size(ids_less_attribute,2)]';
     
     end
     
    end
        
end

answer_tracker_2(useful_id,predicted_label_random_image)=0;

rand_index=randperm(length(unlabeled_ids));

unlabeled_ids=unlabeled_ids(rand_index);

answer_tracker_2=answer_tracker_2(rand_index,:);

negative_examples=negative_examples(rand_index,:);

negative_examples_weights=negative_examples_weights(rand_index,:);

else
    
 variable_train_ids=[variable_train_ids unlabeled_ids(useful_id)];
 
 unlabeled_ids(useful_id)=[];
 
 answer_tracker_2(useful_id,:)=[];
 
 negative_examples(useful_id,:)=[]; % Delete this example from 
 
 %%% the negative examples list
 
 negative_examples_weights(useful_id,:)=[]; % delete the corresponding neagtive weight row   

end


%%%
 
 predicted_probabilities=zeros(length(test_ids)+length(unlabeled_ids),no_class);
 
  %%%%%%%%%%% Normalize the negative example weights
 
 if max(negative_examples_weights(:))>0
 
 negative_examples_weights_normalized=negative_examples_weights...
     ./max(negative_examples_weights(:));
 
 else
     
 negative_examples_weights_normalized=negative_examples_weights;    
     
 end


 
 %%%%%%%%%%%
 
 %%
 
 %%% WE START LEARNING A CLASSIFIER WHENEVER WE HAVE IMAGES FROM TWO
 %%% DIFFERENT CLASSES
 
 %%% In this section we train and test |no_class| one vs all binary classifiers 
 
      present_classes=unique(class_labels(variable_train_ids));
 
 if length(present_classes)>1
 
 for i=1:1:length(present_classes)
     
 variable_class_labels=class_labels(variable_train_ids);
     
 train_ids_positive=variable_train_ids(variable_class_labels(:)==present_classes(i));
 
 train_ids_negative=variable_train_ids(variable_class_labels(:)~=present_classes(i));
 
 no_definite_pos_neg_examples=length(train_ids_positive)+length(train_ids_negative);
 
 extra_negative_examples_id= unlabeled_ids(negative_examples(:,present_classes(i))==1);
 
 if weight_flag==1
 
 temp=negative_examples_weights_normalized(:,present_classes(i));
 
 extra_negative_examples_weight_vector=...
     temp(negative_examples_weights_normalized(:,present_classes(i))~=0);
 
 else
 
 extra_negative_examples_weight_vector=ones(1,size(extra_negative_examples_id,2));    
     
 end
 
 train_ids_negative=[train_ids_negative extra_negative_examples_id];
 
 train_ids=[train_ids_positive train_ids_negative];
 
 %%%% Updating the weight vector file
 
 weight_vector=ones(length(train_ids),1); 
 
 weight_vector(no_definite_pos_neg_examples+1:length(train_ids))=...
     extra_negative_examples_weight_vector;
 
  fid=fopen('weight_file_AAW','w');
  
  fprintf(fid,'%6.2f\n',weight_vector);
  
  fclose(fid);
 %%%
 
 train_labels=[ ones(1,length(train_ids_positive))...
     -1*ones(1,length(train_ids_negative))]';
 
 test_labels=class_labels([test_ids unlabeled_ids]); % needed to test the
 
 %%% classification accuracy for the test data set
 
 train_features= feat(train_ids,:); 
 
 test_features= feat([test_ids unlabeled_ids],:); 
 
 %%% Training_and_testing
     
 model=svmtrain(weight_vector,train_labels,train_features,['-b 1 -s 0 -t 0 -c ' num2str(1000) ' -g ' num2str(4/feat_dim)]);
 
 %%% Now we use the binary classifiers to classify our test samples and try
 %%% to see if they are correct or not
 
 [predicted_label, accuracy, predicted_probabilities_temp] = svmpredict(test_labels, test_features, model ,'-b 1');
 
 predicted_probabilities(:,present_classes(i))=predicted_probabilities_temp(:,1); % We find the positive 
 
 %%% probabilities of an image
 
 end
 
 %%
 
 % Normalization
 
 clear sum_predicted_probabilities;
 
 predicted_probabilities(length(test_ids)+1:length(test_ids)+length(unlabeled_ids),:)=...
     predicted_probabilities(length(test_ids)+1:length(test_ids)+length(unlabeled_ids),:).*answer_tracker_2; 
     
 sum_predicted_probabilities=sum(predicted_probabilities,2);

 sum_predicted_probabilities_repeated=repmat(sum_predicted_probabilities,1,no_class);

 predicted_probabilities=predicted_probabilities./sum_predicted_probabilities_repeated;
 
 [maximum, predicted_labels]=max(predicted_probabilities,[],2); 
 
 predicted_labels_test=predicted_labels(1:length(test_ids));

 predicted_label_difference=test_labels(1:length(test_ids))-predicted_labels_test;
 
 correct_predictions=length(predicted_label_difference(...
     predicted_label_difference(:)==0));
 
 classification_accuracy(nrun,ts)=correct_predictions*100/length(test_ids);
 
 end
 
%  size(relative_attribute_predictor)
%  
%  [correct_percentage_on_the_fly, correct_percentage_pretrained]=...
%      accuracy_attribute_models_function(relative_attribute_predictor,...
%      no_attribute, no_images, no_more_votes, image_indices);
%  
%  store_attribute_accuracy_on_the_fly(ts)=correct_percentage_on_the_fly;
%  
%  store_attribute_accuracy_pretrained(ts)=correct_percentage_pretrained;
 
 end
 
 save(['JCA_ARW_2_w' num2str(weight_flag) '_v' num2str(active_version) '_u' num2str(uniform_flag) '_wa' num2str(weight_active_flag) '.mat'],'classification_accuracy');
 
 end
 
 save(['JCA_ARW_2_w' num2str(weight_flag) '_v' num2str(active_version) '_u' num2str(uniform_flag) '_wa' num2str(weight_active_flag) '.mat'],'classification_accuracy');
 
 %%% Plotting the number of training instances vs accuracy of
 %%% classification of the unalebeled data
 
%  plot(1:1:length(classification_accuracy) ,classification_accuracy,...
%      'LineWidth',4);
%  grid;

save('attribute_accuracy_fly_and_pretrained.mat','store_attribute_accuracy_on_the_fly','store_attribute_accuracy_pretrained');

end




 
