function [image_id, image_id_in_unlabeled_ids] =...
most_informative_image_id_new_1(unlabeled_ids, variable_train_ids,...
predicted_probabilities_permanent,relative_attribute_predictions,no_class,ts,uniform_flag)

%%% In this file we implement active learning to get labels for our
%%% unlabeled dataset. We try to find which one is the most informative or
%%% useful data point to get a label. So for each data point we have to
%%% calculate the expected reduction in entropy

%%% Predicted_probabilities_permanent is the matrix of probabilities of the unlabeled
%%% data only.
 
load data.mat;

clearvars attribute_names relative_att_predictions relative_att_predictor;

load human_attribute_results_1.mat;

relative_att_predictions=relative_attribute_predictions;

%%%% First we calculate the present entropy of the system for all the
%%%% unlabeled data points

prob_matrix=predicted_probabilities_permanent;

log_prob_matrix=log(prob_matrix);

product_matrix=prob_matrix.*log_prob_matrix;

product_matrix(isnan(product_matrix))=0;

present_entropy=sum(product_matrix(:));

%%%%

 relative_attributes_unlabeled_data=relative_att_predictions(unlabeled_ids,:);
 
 uid=1:1:length(unlabeled_ids);
 
 vtd=1:1:length(variable_train_ids);
 
 expected_entropy=zeros(length(unlabeled_ids),1);
 
 expected_change_entropy=zeros(length(unlabeled_ids),1);

for ui=1:1:length(unlabeled_ids)
    
    %%% 'changed_entropy_no' array will store the expected entropy for 2*K possible
    %%% negative results and 'probability_no' array will store corresponding
    %%% probabilities
    
    changed_entropy_no=zeros(2*length(attribute_names),1);
    
    %%% Also find the most probable class for image ui, and also train images of
    %%% class "predicted_label"
    
    [probability_yes, predicted_label]=max(predicted_probabilities_permanent(ui,:)); 
              
     class_labels_vtd=class_labels(variable_train_ids);
          
     train_images_predicted_label=vtd(class_labels_vtd(:)==predicted_label);
     
     if uniform_flag==0
    
        probability_no=((1-probability_yes)./(2*length(attribute_names))).*ones(2*length(attribute_names),1);  
         
     else
         
        probability_no=zeros(2*length(attribute_names),1);
         
     end
    
    %%% For each data point we have total 2*K+1 possible inputs from the
    %%% user (K is the total number of attributes) , 
    %%% Those inputs are {yes}, {no, a_1,too_much},{no,a_1,too_less},
    %%% {no,a_2,too_much},{no,a_2,too_less},...........................
    %%% For all of these 2*K+1 inputs we have to find the probabilities of
    %%% that event
    
    for i=1:1:length(attribute_names)
        
        if sum(relative_attribute_predictions(:,i))~=0
     
     unlabeled_attribute_i=relative_attributes_unlabeled_data(:,i);
     
     img_id_attribute_strength=relative_att_predictions(unlabeled_ids(ui),i);
     
     ids_more_attribute= uid(unlabeled_attribute_i(:)>img_id_attribute_strength);
     
     ids_less_attribute= uid(unlabeled_attribute_i(:)<img_id_attribute_strength);
     
     % Case 1:
     
     predicted_probabilities=predicted_probabilities_permanent;
     
     predicted_probabilities(ids_more_attribute,predicted_label)=0;
     
     % Normalization
     
     sum_predicted_probabilities=sum(predicted_probabilities,2);
     
     sum_predicted_probabilities_repeated=repmat(sum_predicted_probabilities,1,no_class);
     
     predicted_probabilities=predicted_probabilities./sum_predicted_probabilities_repeated;
     
     %%% Calculate the entropy
     
     prob_matrix=predicted_probabilities;
     
     prob_matrix(ui,:)=[]; 

     log_prob_matrix=log(prob_matrix);
     
     product_matrix=prob_matrix.*log_prob_matrix;
     
     product_matrix(isnan(product_matrix))=0;

     changed_entropy_no(2*i-1)=sum(product_matrix(:));
         
     %%%
     
     % Case 2:
     
     predicted_probabilities=predicted_probabilities_permanent;
     
     predicted_probabilities(ids_less_attribute,predicted_label)=0;
     
     % Normalization
     
     sum_predicted_probabilities=sum(predicted_probabilities,2);
     
     sum_predicted_probabilities_repeated=repmat(sum_predicted_probabilities,1,no_class);
     
     predicted_probabilities=predicted_probabilities./sum_predicted_probabilities_repeated;
     
     %%% Calculate the entropy
     
     prob_matrix=predicted_probabilities;
     
     prob_matrix(ui,:)=[];

     log_prob_matrix=log(prob_matrix);
     
     product_matrix=prob_matrix.*log_prob_matrix;
     
     product_matrix(isnan(product_matrix))=0;

     changed_entropy_no(2*i)=sum(product_matrix(:));

    %%% Calculate the probability
    
    if uniform_flag~=0
     
     variable_train_id_attribute_i=...
         relative_attribute_predictions(variable_train_ids...
         (train_images_predicted_label),i);
     
     deviation_attribute=(sum(img_id_attribute_strength-variable_train_id_attribute_i));
     
     if size(variable_train_id_attribute_i,1)>=1
         
         deviation_attribute=deviation_attribute./size(variable_train_id_attribute_i,1);
         
     end
     
     if (max(relative_attribute_predictions(:,i))-min(relative_attribute_predictions(:,i)))>0
         
         deviation_attribute=deviation_attribute./...
             (max(relative_attribute_predictions(:,i))-min(relative_attribute_predictions(:,i)));
         
     end
     
     probability_no(2*i-1)=deviation_attribute;
     
     probability_no(2*i)=-deviation_attribute;
        
    end
    
     else

     changed_entropy_no(2*i-1)=present_entropy; probability_no(2*i-1)=0;

     changed_entropy_no(2*i)=present_entropy; probability_no(2*i)=0;

     end
    
    end
    
    min_prob_no=min(probability_no);
    
    probability_no=probability_no-min_prob_no;
    
    probability_no(probability_no(:)==abs(min_prob_no))=0;
    
    if sum(probability_no)>0
    
        probability_no=probability_no./sum(probability_no);
        
    end
    
    probability_no=abs(probability_no);

    %%% Normalizing the negative probabilities
    
    sum_probability_no=sum(probability_no);
    
    if sum_probability_no>0 
    
    probability_no=(probability_no.*(1-probability_yes))./sum_probability_no;   
    
    end
    
    %%% Find entropy if the answer is positive
    
    possible_unlabeled_ids=1:1:length(unlabeled_ids);
    
    possible_unlabeled_ids(ui)=[];
    
    prob_matrix=predicted_probabilities_permanent(possible_unlabeled_ids,:);
    
    log_prob_matrix=log(prob_matrix);
    
    product_matrix=prob_matrix.*log_prob_matrix;
    
    product_matrix(isnan(product_matrix))=0;
    
    changed_entropy_yes=sum(product_matrix(:));
    
    expected_entropy(ui)=probability_yes*changed_entropy_yes+...
        sum(probability_no.*changed_entropy_no);
    
    expected_change_entropy(ui)=present_entropy-expected_entropy(ui);
    
    %%%% Stopping code for debugging
    
end

%%% Now we have to find the data point for which the expected reduction in
%%% entropy will be maximized

[maximum,max_index]=max(abs(expected_change_entropy));

image_id_in_unlabeled_ids=max_index;

image_id=unlabeled_ids(max_index);

%%% The above id is the image id which we think should give us maximum
%%% benifit in terms of a user input
