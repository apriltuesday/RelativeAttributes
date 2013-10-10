function [image_id, image_id_in_unlabeled_ids] =...
most_informative_image_id_joint_new_fast_6(unlabeled_ids, variable_train_ids,...
predicted_probabilities_permanent,relative_attribute_predictions_now,relative_attribute_predictor_now,no_class,...
attribute_based_feedback,no_attr_based_feedback,uniform_flag)

tic

%%% In this file we implement active learning to get labels for our
%%% unlabeled dataset. We try to find which one is the most informative or
%%% useful data point to get a label. So for each data point we have to
%%% calculate the expected reduction in entropy

%%% Predicted_probabilities_permanent is the matrix of probabilities of the unlabeled
%%% data only.

load data.mat;

load human_attribute_results_1.mat;

no_images=length(im_names);

%%%% First we calculate the present entropy of the system for all the
%%%% unlabeled data points

prob_matrix=predicted_probabilities_permanent;

log_prob_matrix=log(prob_matrix);

product_matrix=prob_matrix.*log_prob_matrix;

product_matrix(isnan(product_matrix))=0;

present_entropy=sum(product_matrix(:));

%%%%
 
 uid=1:1:length(unlabeled_ids);
 
 expected_entropy=zeros(length(unlabeled_ids),1);
 
 expected_change_entropy=zeros(length(unlabeled_ids),1);
 
   %% initilize the preference matrices 
 
 max_no_inequality_preference=2000; no_inequality_preference=ones(length(attribute_names),1);
 
 max_no_equality_preference=0; no_equality_preference=ones(length(attribute_names),1);
 
 inequality_preference_matrix=zeros(length(attribute_names),...
     max_no_inequality_preference,size(feat,1));
 
 equality_preference_matrix=zeros(length(attribute_names),...
     max_no_equality_preference,size(feat,1));
 
 %% Introduce the attribute based feedback to those matrices
 
 for i=1:1:no_attr_based_feedback
     
   inequality_preference_matrix(attribute_based_feedback(i,3),...
       no_inequality_preference(attribute_based_feedback(i,3)),...
        attribute_based_feedback(i,1))=-1;

   inequality_preference_matrix(attribute_based_feedback(i,3),...
       no_inequality_preference(attribute_based_feedback(i,3)),...
        attribute_based_feedback(i,2))=+1;

    no_inequality_preference(attribute_based_feedback(i,3))=...
        no_inequality_preference(attribute_based_feedback(i,3))+1;
          
 
 end
 
 %% DEFINE DATA STRUCTURES FOR THIS FASTER METHOD
 
 max_num_v_constraints=5000;
 
 cost_per_attribute=25; % We set this parameter based on how much time we want to spend per attribute
 
 probability_yes_all=zeros(length(unlabeled_ids),1); predicted_label_all=zeros(length(unlabeled_ids),1);
 
 probability_no_all=zeros(length(unlabeled_ids),2*length(attribute_names));
 
 train_images_predicted_label_all=zeros(no_class,100); % Max number of elements per class is 100
 
     for ui=1:1:length(unlabeled_ids)

     [probability_yes_all(ui), predicted_label_all(ui)]=max(predicted_probabilities_permanent(ui,:)); 

     end
 
     for nc=1:1:no_class

         temp=variable_train_ids(class_labels(variable_train_ids)==nc);

         train_images_predicted_label_all(nc,1:size(temp,2))=temp;

     end
     
     %% SOME MORE VARIABLES!
     
     relative_attribute_predictions_all=zeros(cost_per_attribute,no_images,length(attribute_names));
     
     changed_entropy_no=zeros(length(unlabeled_ids),2*length(attribute_names));
     
     changed_entropy_yes=zeros(length(unlabeled_ids),1);
 
 %% STARTING THE FOR LOOPS
 
 for i=1:1:length(attribute_names)
     
 %%  TOO OPEN CASE:
      
 data_to_constraint_pointer_1=zeros(length(unlabeled_ids),1); % We only take care of the maximum violating constraint
 
 num_of_violating_constraints_1=zeros(length(unlabeled_ids),1);
 
 %% NOT OPEN ENOUGH CASE
 
 data_to_constraint_pointer_2=zeros(length(unlabeled_ids),1);
 
 num_of_violating_constraints_2=zeros(length(unlabeled_ids),1);
 
 %% OTHER DATA STUCTURES
 
 all_violating_constraints=zeros(max_num_v_constraints,2); 
 
 num_all_violating_constraints=0;
 
 %%
     
     %%% First collect all the constraints, which are violated for the i-th attribute
     
     for ui=1:1:length(unlabeled_ids)
          
     train_images_predicted_label=train_images_predicted_label_all(predicted_label_all(ui),:);
     
     train_images_predicted_label=train_images_predicted_label(train_images_predicted_label(:)~=0);
     
     %% CASE 1: When expected input from user "This image is too open to be forest"
     
     tipl_vc= train_images_predicted_label(relative_attribute_predictions_now(train_images_predicted_label,i)...
         >relative_attribute_predictions_now(unlabeled_ids(ui),i));
     
     if  ~isempty(tipl_vc)
         
     [~,max_index]=max(relative_attribute_predictions_now(tipl_vc,i)); 
     
     tipl_vc=tipl_vc(max_index);% Taking only the maximum violated training_image
         
     all_violating_constraints(num_all_violating_constraints+1,:)=...
         [tipl_vc unlabeled_ids(ui)];
     
     data_to_constraint_pointer_1(ui,1)=num_all_violating_constraints+1;     
          
     num_all_violating_constraints=num_all_violating_constraints+1;
     
     num_of_violating_constraints_1(ui)=1;
     
     end
     
      %% CASE 2: When expected input from user "This image is not open enough to be forest"
      
     tipl_vc= train_images_predicted_label(relative_attribute_predictions_now(train_images_predicted_label,i)...
         <relative_attribute_predictions_now(unlabeled_ids(ui),i));
     
     if ~isempty(tipl_vc)
         
     [~,min_index]=min(relative_attribute_predictions_now(tipl_vc,i)); 
     
     tipl_vc=tipl_vc(min_index);% Taking only the maximum violated training_image

     all_violating_constraints(num_all_violating_constraints+1,:)=[unlabeled_ids(ui) tipl_vc];
     
     data_to_constraint_pointer_2(ui,1)=num_all_violating_constraints+1;     
          
     num_all_violating_constraints=num_all_violating_constraints+1;
     
     num_of_violating_constraints_2(ui)=1;
     
     end
     
     end
     
     %% NOW WE HAVE TO FIND UNIQUE PAIRS FROM ALL THE VIOLATED CONSTRAINTS and THEN CLUSTERING!
     
     [unique_violating_constraints,ia,ic]=unique(all_violating_constraints(1:num_all_violating_constraints,:),'rows');
     
     unique_violating_constraints_features=feat(unique_violating_constraints(:,1),:)-feat(unique_violating_constraints(:,2),:);
     
     %% IF LOOP TO SEE IF THERE ARE AT ALL ANY VIOLATING CONSTRAINTS
     
     if ~isempty(unique_violating_constraints_features)
     
     [label, ~, index] = kmedoids(unique_violating_constraints_features',cost_per_attribute);
     
     %% NOW WE FIND THE NEW RANKING FUNCTIONS AND RANKS FOR THE CLUSTERED PAIRS
     
         for no_clusters=1:1:cost_per_attribute

         mat_temp=reshape(inequality_preference_matrix(i,:,:),[max_no_inequality_preference no_images]);

         no_inequality_preference_temp=no_inequality_preference;
       
          mat_temp(no_inequality_preference(i),unique_violating_constraints(index(no_clusters),1))=-1;

          mat_temp(no_inequality_preference(i),unique_violating_constraints(index(no_clusters),2))=+1;

          no_inequality_preference_temp(i)=no_inequality_preference_temp(i)+1;

            [relative_attribute_predictor, relative_attribute_predictions_all(no_clusters,:,:)]=...
         learn_the_attribute_model_for_single_attribute(mat_temp,no_inequality_preference_temp,...
         equality_preference_matrix,no_equality_preference,relative_attribute_predictor_now,...
         relative_attribute_predictions_now,i,feat);

         end
         
     %% FINALLY WE HAVE TO EVALUATE THE CHANGE IN ENTROPY FOR ALL POINTS FOR NEGATIVE REPLIES
        
     for ui=1:1:length(unlabeled_ids)
         
         %% CASE 1: 
         
         if num_of_violating_constraints_1(ui)~=0
         
         relative_attribute_predictions=...
             reshape(relative_attribute_predictions_all(label(ic(data_to_constraint_pointer_1(ui,1))),:,:),...
             size(relative_attribute_predictions_now)); 
         
         else
             
         relative_attribute_predictions=relative_attribute_predictions_now;
             
         end
         
         relative_attributes_unlabeled_data=relative_attribute_predictions(unlabeled_ids,:);
     
         unlabeled_attribute_i=relative_attributes_unlabeled_data(:,i);

         img_id_attribute_strength=relative_attribute_predictions(unlabeled_ids(ui),i);

         ids_more_attribute= uid(unlabeled_attribute_i(:)>img_id_attribute_strength);

         predicted_probabilities=predicted_probabilities_permanent;

         predicted_probabilities(ids_more_attribute,predicted_label_all(ui))=0;

         % Normalization

         sum_predicted_probabilities=sum(predicted_probabilities,2);

         sum_predicted_probabilities_repeated=repmat(sum_predicted_probabilities,1,no_class);

         predicted_probabilities=predicted_probabilities./sum_predicted_probabilities_repeated;

         %%% Calculate the entropy

         prob_matrix=predicted_probabilities;

         log_prob_matrix=log(prob_matrix);

         product_matrix=prob_matrix.*log_prob_matrix;

         product_matrix(isnan(product_matrix))=0;

         changed_entropy_no(ui,2*i-1)=sum(product_matrix(:));
         
     %%%
         
        %% CASE 2: 
         
         if num_of_violating_constraints_2(ui)~=0
         
         relative_attribute_predictions=...
             reshape(relative_attribute_predictions_all(label(ic(data_to_constraint_pointer_2(ui,1))),:,:),...
             size(relative_attribute_predictions_now));
         
         else
             
         relative_attribute_predictions=relative_attribute_predictions_now;
             
         end
         
         relative_attributes_unlabeled_data=relative_attribute_predictions(unlabeled_ids,:);
     
         unlabeled_attribute_i=relative_attributes_unlabeled_data(:,i);

         img_id_attribute_strength=relative_attribute_predictions(unlabeled_ids(ui),i);

         ids_less_attribute= uid(unlabeled_attribute_i(:)<img_id_attribute_strength);

         predicted_probabilities=predicted_probabilities_permanent;

         predicted_probabilities(ids_less_attribute,predicted_label_all(ui))=0;

         % Normalization

         sum_predicted_probabilities=sum(predicted_probabilities,2);

         sum_predicted_probabilities_repeated=repmat(sum_predicted_probabilities,1,no_class);

         predicted_probabilities=predicted_probabilities./sum_predicted_probabilities_repeated;

         %%% Calculate the entropy

         prob_matrix=predicted_probabilities;

         log_prob_matrix=log(prob_matrix);

         product_matrix=prob_matrix.*log_prob_matrix;

         product_matrix(isnan(product_matrix))=0;

         changed_entropy_no(ui,2*i)=sum(product_matrix(:));
         
     %%% Calculating the probabilities
     if sum(relative_attribute_predictions(:,i))~=0
         
         train_images_predicted_label=train_images_predicted_label_all(predicted_label_all(ui),:);
     
         train_images_predicted_label=train_images_predicted_label(train_images_predicted_label(:)~=0);
     
             if uniform_flag~=0
     
                 variable_train_id_attribute_i=...
                     relative_attribute_predictions(train_images_predicted_label,i);

                 deviation_attribute=(sum(img_id_attribute_strength-variable_train_id_attribute_i));
     
                 if size(variable_train_id_attribute_i,2)>=1

                     deviation_attribute=deviation_attribute./size(variable_train_id_attribute_i,1);

                 end

                 if (max(relative_attribute_predictions(:,i))-min(relative_attribute_predictions(:,i)))>0

                     deviation_attribute=deviation_attribute./...
                         (max(relative_attribute_predictions(:,i))-min(relative_attribute_predictions(:,i)));

                 end

                 probability_no_all(ui,2*i-1)=deviation_attribute;

                 probability_no_all(ui,2*i)=-deviation_attribute;      

            end
     
     else

     probability_no_all(ui,2*i-1)=0;

     probability_no_all(ui,2*i)=0;

     end  
         
     end
     
     end
     
 end
 
 %% FINDING THE CHANGE IN ENTROPY FOR POSITIVE ANSWERS AND ALSO EXPECTED CHANGE IN ENTROPY
 
 for ui=1:1:length(unlabeled_ids)  
     
     %%
     
     if uniform_flag==0
    
        probability_no=(((1-probability_yes_all(ui))./(2*length(attribute_names))).*ones(2*length(attribute_names),1))';  
         
     else
         
        probability_no=probability_no_all(ui,:);
        
        min_prob_no=min(probability_no);
    
        probability_no=probability_no-min_prob_no;

        probability_no(probability_no(:)==abs(min_prob_no))=0;
        
        sum_probability_no=sum(probability_no);

        if sum(probability_no)>0

           probability_no=(probability_no.*(1-probability_yes_all(ui)))./sum_probability_no;  

        end

        probability_no=abs(probability_no);
        
     end

     %%
 
    possible_unlabeled_ids=1:1:length(unlabeled_ids);
    
    possible_unlabeled_ids(ui)=[];
    
    prob_matrix=predicted_probabilities_permanent(possible_unlabeled_ids,:);
    
    log_prob_matrix=log(prob_matrix);
    
    product_matrix=prob_matrix.*log_prob_matrix;
    
    product_matrix(isnan(product_matrix))=0;
    
    changed_entropy_yes(ui)=sum(product_matrix(:));
    
    probability_yes=probability_yes_all(ui);
    
    expected_entropy(ui)=probability_yes*changed_entropy_yes(ui)+...
        sum(probability_no.*changed_entropy_no(ui,:));

    expected_change_entropy(ui)=present_entropy-expected_entropy(ui);
    
 end

%%% Now we have to find the data point for which the expected reduction in
%%% entropy will be maximized

[maximum,max_index]=max(abs(expected_change_entropy));

image_id_in_unlabeled_ids=max_index;

image_id=unlabeled_ids(max_index);

toc

%%% The above id is the image id which we think should give us maximum
%%% benifit in terms of a user input
