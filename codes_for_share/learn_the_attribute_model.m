function [relative_attribute_predictor, relative_attribute_predictions]=...
     learn_the_attribute_model(attribute_based_feedback,no_attr_based_feedback)
 
 % In attribute_based_feedback, the second element has always the more
 % attribute than the first element
 
 load data.mat; feat_dim=size(feat,2);
 
 load human_attribute_results_1.mat; 
 
 
 %% Initilize the return variables
 
  relative_attribute_predictor=zeros(size(feat,2),length(attribute_names));
  
  relative_attribute_predictions=zeros(size(feat,1),length(attribute_names));
  
 if no_attr_based_feedback>0
 
  %% initilize the preference matrices
 
 max_no_inequality_preference=2000; no_inequality_preference=ones(length(attribute_names),1);
 
 max_no_equality_preference=0; no_equality_preference=ones(length(attribute_names),1);
 
 inequality_preference_matrix=zeros(length(attribute_names),...
     max_no_inequality_preference,size(feat,1));
 
 equality_preference_matrix=zeros(length(attribute_names),...
     max_no_equality_preference,size(feat,1));
 
 %% Introduce the attribute based feedback to those matrices
 
 for i=1:1:no_attr_based_feedback
     
     try
     
   inequality_preference_matrix(attribute_based_feedback(i,3),...
       no_inequality_preference(attribute_based_feedback(i,3)),...
        attribute_based_feedback(i,1))=-1;
    
     catch
         
        % keyboard
         
     end

   inequality_preference_matrix(attribute_based_feedback(i,3),...
       no_inequality_preference(attribute_based_feedback(i,3)),...
        attribute_based_feedback(i,2))=+1;

    no_inequality_preference(attribute_based_feedback(i,3))=...
        no_inequality_preference(attribute_based_feedback(i,3))+1;
          
 
 end
 
 %% Finally learning the attribute functions
 
 for no_attr=1:1:length(attribute_names)
     
      training_error_penalization=ones(no_inequality_preference(no_attr)-1,1).*0.1;
      
       w=zeros(feat_dim,1);  
       
      if no_inequality_preference(no_attr)>1
       
       sparse_inequality_preference_matrix=...
           sparse(reshape(inequality_preference_matrix(no_attr,...
           1:no_inequality_preference(no_attr)-1,:),...
           [no_inequality_preference(no_attr)-1 size(feat,1)]));
       
       sparse_equality_preference_matrix=...
           sparse(reshape(equality_preference_matrix(no_attr,...
           1:no_equality_preference(no_attr)-1,:),...
           [no_equality_preference(no_attr)-1 size(feat,1)]));
       
       %keyboard

       w=ranksvm_with_sim(feat,sparse_inequality_preference_matrix,...
             sparse_equality_preference_matrix,training_error_penalization, w);
         
      end
         
         relative_attribute_predictor(:,no_attr)=w;     
     
 end

 relative_attribute_predictions=feat*relative_attribute_predictor;
 
 end
 

end