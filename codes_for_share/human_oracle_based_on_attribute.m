function [correct_index, attribute_index,much_or_less_index] =...
human_oracle_based_on_attribute(image_id, predicted_label)

%%% In this file we try to predict if the predicted label is correct and
%%% if not correct, based on the category level feedback on pubfig dataset
%%% from humans we try to provide some attribute based feedback for the
%%% incorrect prediction
 
load data.mat;

load human_attribute_results_1.mat;

if class_labels(image_id)==predicted_label
    
    correct_index=1;
    
    attribute_index=0;
    
    much_or_less_index=0;
    
else

    correct_index=0;
    
        %%% Since our matrix is structured in a special way determining the
        %%% minimum and maximum class is important
        
        min_class=min(class_labels(image_id),predicted_label);
        
        max_class=max(class_labels(image_id),predicted_label);
        
        attribute_differnce=zeros(length(attribute_names),1);
        
        attribute_differnce=attribute_compare(min_class,max_class,:,1)-...
        attribute_compare(min_class,max_class,:,2);
    
        max_diff=max(abs(attribute_differnce));
        
        ad=1:1:length(attribute_differnce);
        
        temp=ad(abs(attribute_differnce)==max_diff); rand_temp=randperm(length(temp));
        
        temp=temp(rand_temp); max_diff_index=temp(1);
        
        attribute_index=max_diff_index;
        
        % Now we have to find the much_or_less index
        
        if min_class==class_labels(image_id) && ...
                attribute_differnce(max_diff_index)>=0 ||...
                max_class==class_labels(image_id) && ...
                attribute_differnce(max_diff_index)<=0
            
            much_or_less_index=+1;
            
        elseif max_class==class_labels(image_id) && ...
                    attribute_differnce(max_diff_index)>=0 ||...
                    min_class==class_labels(image_id) && ...
                    attribute_differnce(max_diff_index)<=0
                
            much_or_less_index=-1;
            
        end

end

end

