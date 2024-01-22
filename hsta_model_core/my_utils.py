# obtainlabelfile
import os
import pandas as pd
import PIL
from PIL import Image
import torch
import numpy as np
import random
# Some parts of the stupid datasetoffset<onset,as


def get_casme3_label(args=None,mode=None):
    
    
    root_path="YOURPATH/casme3_data/wash_new_all/new_labels"
    
    if args.use_emothion_or_objective_class_as_label=="emothion":
        
        if(args.k_folds==0):
            if(args.nb_classes==7):
                print("label position YOURPATH/casme3_data/wash_new_all/new_labels/emothion_as_label/combined_csv.csv")
                return "YOURPATH/casme3_data/wash_new_all/new_labels/emothion_as_label/combined_csv.csv"
            
        what_as_label="emothion_as_label"
        if mode=='train':
            file_name=f'train_fold_{args.k_counts+1}.csv'
            
        elif mode=='test' or mode=='validation':
            file_name=f'test_fold_{args.k_counts+1}.csv'
        
    elif args.use_emothion_or_objective_class_as_label=="objective_class":
        what_as_label="Objective_class_as_label"
        if mode=='train':
            if args.pretrain_on_macro==True:
                file_name="only_Macro_train_dataset.csv"
                print(" pretain macro ###############################################")
            elif args.use_extra_macro_data==True:
                file_name=f'use_macro_exp_train_fold_{args.k_counts+1}.csv'
            elif args.use_extra_macro_data==False:
                file_name=f'no_macro_exp_train_fold_{args.k_counts+1}.csv'
                
            
        elif mode=='test' or mode=='validation':
            if args.pretrain_on_macro==True:
                file_name="only_Macro_test_dataset.csv"
                print(" pretain macro ###############################################")
            else:
                file_name=f'test_fold_{args.k_counts+1}.csv'
            
    label_path=os.path.join(root_path,what_as_label,file_name)
    
    return label_path
    
    
    # objective：Input thelabelReplace with the directory containing the video image
    # conversionlabelvalue，Convert string typehappyWaiting to become1，2，3
    
def clean_data_for_casme3(df=None,args=None,root="YOURPATH/casme3_data/wash_new_all/data"):
    df_new=df.copy()
    df_new['file_path']=[None] * len(df)
    df_new['new_label']=[None] * len(df)
    for index in range(len(df)):
        path=os.path.join(df["Subject"][index],df["Filename"][index],"color")
        path=os.path.join(root,path)
        if os.path.exists(path):
            pass
            
        else:
            print(path,"not found")
            # df_new['file_path'][index]=path
        df_new.loc[index, 'file_path'] = path
    if(args.use_emothion_or_objective_class_as_label=="emothion"):
        label_mapping = {
            'others': 0,
            'Others': 0,
            'disgust': 1,
            'surprise': 2,
            'fear': 3,
            'anger': 4,
            'sad': 5,
            'happy': 6,
            # Add mapping relationships for other emotions
        }
        
    elif args.use_emothion_or_objective_class_as_label=="objective_class":
            label_mapping = {
            'I': 0,
            'II': 1,
            'III': 2,
            'IV': 3,
            'V': 4,
            'VI': 5,
            'VII': 6,
            # Add mapping relationships for other emotions
        }
        
    
    df_new['new_label']=df_new["emotion" if args.use_emothion_or_objective_class_as_label=="emothion" else "Objective class"].map(label_mapping)


    return df_new


def loso_clean_data_for_casme3(df=None,args=None,root="YOURPATH/casme3_data/wash_new_all/data",mode=None):
    df_new=df.copy()
    df_new['file_path']=[None] * len(df)
    df_new['new_label']=[None] * len(df)
    for index in range(len(df)):
        path=os.path.join(df["Subject"][index],df["Filename"][index],"color")
        path=os.path.join(root,path)
        if os.path.exists(path):
            pass
            
        else:
            print(path,"not found")
            # df_new['file_path'][index]=path
        df_new.loc[index, 'file_path'] = path
    if(args.use_emothion_or_objective_class_as_label=="emothion"):
        label_mapping = {
            'others': 0,
            'Others': 0,
            'disgust': 1,
            'surprise': 2,
            'fear': 3,
            'anger': 4,
            'sad': 5,
            'happy': 6,
            # Add mapping relationships for other emotions
        }
        
    elif args.use_emothion_or_objective_class_as_label=="objective_class":
            label_mapping = {
            'I': 0,
            'II': 1,
            'III': 2,
            'IV': 3,
            'V': 4,
            'VI': 5,
            'VII': 6,
            # Add mapping relationships for other emotions
        }
        
    
    df_new['new_label']=df_new["emotion" if args.use_emothion_or_objective_class_as_label=="emothion" else "Objective class"].map(label_mapping)
    
    # First, calculate the totalsub_list
    # Find what is neededloso list，useloso_control_numTaking the remainder of the index yieldsloso list now
    # At this time, useloso_pos_countRecord current progressloso_list_nowWhich position of
    # args.len_of_loso_listRecord total length
    # If during the training phase，Excludeloso list
    # If in other stages, only useloso list
    subjects_list = df_new['Subject'].unique().tolist()
    
    print(subjects_list)

    
    loso_list_now=[elem for i, elem in enumerate(subjects_list) if i % args.sum_num_multidevice_task == args.loso_control_num]
    print(loso_list_now)
    print("now#######################subject:",loso_list_now[args.loso_pos_count])
    args.len_of_loso_list=len(loso_list_now)
    if mode in ['train']:

        filtered_df = df_new[df_new['Subject'] != loso_list_now[args.loso_pos_count]].reset_index(drop=True)
                # Add noise
        if args.no_noise==False:
            print("add noise ###########################################  ")
            print("before add noise len dataset ###########################################         ",len(filtered_df))
            condition = (df_new['Subject'] == loso_list_now[args.loso_pos_count])

            selected_row = df_new[condition].sample(n=1).copy()
            print("selected_row ###########################################",selected_row)
            macro=pd.read_csv("YOURPATH/casme3_data/wash_new_all/new_labels/Objective_class_as_label/Macro-expression.csv")
            macro_matching_rows = macro[macro["Subject"] == loso_list_now[args.loso_pos_count]]


            for i in range(args.nb_classes):
                selected_row_macro=macro_matching_rows.sample()
                selected_row['Apex'],selected_row['Onset'],selected_row['Offset']=selected_row_macro['Apex'].iloc[0],selected_row_macro['Onset'].iloc[0],selected_row_macro['Offset'].iloc[0]
                path=os.path.join(root,selected_row_macro["Subject"].iloc[0],selected_row_macro["Filename"].iloc[0],"color")
                if os.path.exists(path):
                        pass
            
                else:
                    print(path,"not found")
                selected_row['file_path']= path
                selected_row['new_label']=i
                filtered_df=filtered_df._append(selected_row, ignore_index=True)
                
            filtered_df.reset_index(drop=True)
            print("len dataset ###########################################         ",len(filtered_df))
    else:
    # Only takeloso
        filtered_df = df_new[df_new['Subject'] == loso_list_now[args.loso_pos_count]].reset_index(drop=True)


    return filtered_df 

def loso_clean_data_for_smic(df=None,args=None,root="YOURPATH/casme3_data/wash_new_all/data",mode=None):
    df_new=df.copy()

    df_new['file_path']=df["file_path"]
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 100)

    
    df_new['new_label']=df["new_label"]
    
    # First, calculate the totalsub_list
    # Find what is neededloso list，useloso_control_numTaking the remainder of the index yieldsloso list now
    # At this time, useloso_pos_countRecord current progressloso_list_nowWhich position of
    # args.len_of_loso_listRecord total length
    # If during the training phase，Excludeloso list
    # If in other stages, only useloso list
    subjects_list = df_new['subject'].unique().tolist()
    print(subjects_list)
    loso_list_now=[elem for i, elem in enumerate(subjects_list) if i % args.sum_num_multidevice_task == args.loso_control_num]
    print("now#######################subject:",loso_list_now[args.loso_pos_count])
    args.len_of_loso_list=len(loso_list_now)
    if mode in ['train']:

        filtered_df = df_new[df_new['subject'] != loso_list_now[args.loso_pos_count]].reset_index(drop=True)
            # Add noise
        if args.no_noise==False:
            print("add noise ###########################################  ")
            print("before add noise len dataset ###########################################         ",len(filtered_df))
            condition = (df_new['subject'] == loso_list_now[args.loso_pos_count])

            selected_row = df_new[condition].sample(n=1).copy()
            print("selected_row ###########################################",selected_row)
            selected_row['Apex'] = selected_row['Offset']
            selected_row['Onset'] = selected_row['Offset']

            for i in range(args.nb_classes):
                selected_row['new_label']=i
                filtered_df=filtered_df._append(selected_row, ignore_index=True)
                
            filtered_df.reset_index(drop=True)
            print("len dataset ###########################################         ",len(filtered_df))
    else:
    # Only takeloso
        filtered_df = df_new[df_new['subject'] == loso_list_now[args.loso_pos_count]].reset_index(drop=True)
 
  
    # filtered_df.to_csv("YOURPATH/me_data/temp_check_samm_label")

    return filtered_df






# Modifying Reading Videos，And the frame rate is not enough to make up for it0，Excess cropping
# When usingusta，Output as16Frame Before2The frames are the start andapex
# When usingcross，Output as18Frame Before2The frames are the start andapex
def new_load_video_from_path(sample=None,index=None,df=None,args=None):
    from PIL import Image
    Apex_num=df['Apex'][index].astype(int)
    Onset_num=df['Onset'][index].astype(int)
    Offset_num=df['Offset'][index].astype(int)
    # Solving Foolish DatasetsapexMay be smaller thanOnsetProblem with
    if Apex_num<Onset_num:
        Apex_num=Onset_num
    if Offset_num<Onset_num:
        Offset_num=Onset_num        
    if Apex_num>Offset_num:
        Apex_num=(Onset_num + Offset_num )//2
        
    # Solving Foolish DatasetsoffsetMay be smaller thanOnsetProblem with,If the video length is0，Go straight to completion0
    if args.data_set=='smic':
        Apex_path=os.path.join(sample,"image"+str(Apex_num).zfill(6)+'.jpg')
        Onset_path=os.path.join(sample,"image"+str(Onset_num).zfill(6)+'.jpg')
    elif args.data_set=='samm':
        files = [f for f in os.listdir(sample)]
        first_file = files[0]
        file_parts = first_file.split('_')
        number_part = file_parts[1].split('.jpg')[0]
        number_part_length = len(number_part)
        Apex_path=os.path.join(sample,str(df["subject"][index]).zfill(3)+"_"+str(Apex_num).zfill(number_part_length)+'.jpg')
        Onset_path=os.path.join(sample,str(df["subject"][index]).zfill(3)+"_"+str(Onset_num).zfill(number_part_length)+'.jpg')
    elif args.data_set=='casme2':
        Apex_path=os.path.join(sample,"reg_img"+str(Apex_num)+'.jpg')
        Onset_path=os.path.join(sample,"reg_img"+str(Onset_num)+'.jpg')
    else:
        Apex_path=os.path.join(sample,str(Apex_num)+'.jpg')
        Onset_path=os.path.join(sample,str(Onset_num)+'.jpg')
    loaded_frames = []
    width=512
    height=512
    width, height = Image.open(Apex_path).size
    if "cross" in args.model :
        frame_limit=args.num_frames
    else:
        if args.not_put_apex_at_last_for_usta==True:
            frame_limit=args.num_frames
        else:
            frame_limit=args.num_frames-2
    
    # frame_limit=16 if "cross" in args.model else 14
    # A silly dataset may have thousands of videos per segment
    Offset_num=min(Offset_num,Onset_num+50)
    if args.data_set=='smic':
        offset_path=os.path.join(sample,"image"+str(Offset_num).zfill(6)+'.jpg')
    elif args.data_set=='samm':
        offset_path=os.path.join(sample,str(df["subject"][index]).zfill(3)+"_"+str(Offset_num).zfill(number_part_length)+'.jpg')
    elif args.data_set=='casme2':
        offset_path=os.path.join(sample,"reg_img"+str(Offset_num)+'.jpg')
    else:
        offset_path=os.path.join(sample,str(Offset_num)+'.jpg')

    
    if "exp4"in args.model or args.lv_ls<0 :
        if Onset_num>Offset_num:#Prevent Foolish Dataset Mishandling
            Offset_num=Onset_num+2
        first_number=second_number=0
        if args.small_frame_choice==1:
            first_number, second_number = random.sample(range(Onset_num, Offset_num + 1), 2)

            # Ensure small numbers come first
            if first_number > second_number:
                first_number, second_number = second_number, first_number
        if args.small_frame_choice==2:
            first_number, second_number =Onset_num, Apex_num

        if args.small_frame_choice==3:
            first_number, second_number =Apex_num, Offset_num        
        frame_path1 = os.path.join(sample, f"{first_number}.jpg")
        frame_path2= os.path.join(sample, f"{second_number}.jpg")
        if os.path.exists(frame_path1) and os.path.exists(frame_path2):
            frame=Image.open(frame_path1)
            loaded_frames.append(frame)
            frame=Image.open(frame_path2)
            loaded_frames.append(frame)
        else:
            zero_frame = Image.new('RGB', (width, height), 0)  # Assuming a black image
            loaded_frames.append(zero_frame)
            loaded_frames.append(zero_frame)
        sample_test_video = torch.stack([torch.from_numpy(np.array(frame)) for frame in loaded_frames])  
        sample_test_video = sample_test_video.permute(0, 3, 1, 2)

        # Adjust the dimension order to(num_frames,num_channels,  height, width)
        return sample_test_video
          
        
    for i in range(Onset_num, Offset_num + 1):
        if args.data_set=='smic':
            frame_path = os.path.join(sample, "image"+str(i).zfill(6)+".jpg")
        elif args.data_set=='samm':
            frame_path=os.path.join(sample,str(df["subject"][index]).zfill(3)+"_"+str(i).zfill(number_part_length)+'.jpg')           
        elif args.data_set=='casme2':
            frame_path=os.path.join(sample,"reg_img"+str(i)+'.jpg')
        else:
            frame_path = os.path.join(sample, f"{i}.jpg")
        if os.path.exists(frame_path):
            frame=Image.open(frame_path)
            loaded_frames.append(frame)
            # image = cv2.imread(frame_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert toRGBformat
            # loaded_frame  = PIL.Image.fromarray(image)  # Convert toPILImage format
            # loaded_frames.append(loaded_frame)
        else:
            print(f"######################Frame file missing: {frame_path}#############################")
    # Adjust the number of frames
    num_frames = len(loaded_frames)
    if num_frames >= frame_limit:
        # stayloaded_framesRandomly select from the listframe_limitElements，同时保持它们stay原列表中的相对顺序不变
        selected_indices = sorted(random.sample(range(num_frames), frame_limit))
        loaded_frames = [loaded_frames[i] for i in selected_indices]

    elif num_frames < frame_limit:
        # if (num_frames==0):
        #     print(sample,Onset_num, Offset_num + 1,"have problem!!!")
        #     raise FileNotFoundError(f"have problem!!!: {sample,Onset_num, Offset_num}")
        # Fill the remaining frames with zeros
        if args.fill_by_zeros_or_img=="zeros" or num_frames==0:
            # Fill the remaining frames with zeros
            zero_frame = Image.new('RGB', (width, height), 0)  # Assuming a black image
            loaded_frames += [zero_frame] * (frame_limit - num_frames)
        elif args.fill_by_zeros_or_img=="img":
            # Calculate how many frames to add on each side
            frames_to_add = frame_limit - num_frames
            half_frames_to_add = frames_to_add // 2

            # Get the first and last frame

            first_frame = loaded_frames[0]
            last_frame = loaded_frames[-1]

            # Add frames to the beginning and end
            loaded_frames = [first_frame] * half_frames_to_add + loaded_frames + [last_frame] * half_frames_to_add

            # If an odd number of frames is to be added, add one more frame to the end
            if frames_to_add % 2 != 0:
                loaded_frames.append(last_frame)
                
    
    imageonset = Image.open(Onset_path)
    if imageonset==None:
        imageonset=Image.new('RGB', (width, height), 0)
    imageapex = Image.open(Apex_path)
    if imageapex ==None:
        imageapex =Image.new('RGB', (width, height), 0) 
        
    imageoffset = Image.open(offset_path)
    if imageoffset ==None:
        imageoffset =Image.new('RGB', (width, height), 0)      
    # Finally readapex onset
    if (args.use_optflow!=0):
        img1,img2=getoptflow(Onset_path,Apex_path,offset_path,args)
        loaded_frames.append(img1) 
        loaded_frames.append(img2) 
        
    else:
        loaded_frames.append(imageonset) 
        loaded_frames.append(imageapex) 
    



            
    # desired_size = (512, 512)
    # loaded_frames = [frame.resize(desired_size) for frame in loaded_frames]
    if args.data_set=='samm':
        loaded_frames=[frame.convert("RGB") for frame in loaded_frames]
        
    sample_test_video = torch.stack([torch.from_numpy(np.array(frame)) for frame in loaded_frames])  

    # (num_frames, height, width, num_channels)

    # shape=pd.DataFrame()
    # shape["shape"]=sample_test_video.shape
    # shape['len loaded_frames']=len(loaded_frames)
    # shape.to_csv("YOURPATH/me_data/temp_check_samm_shape")

    sample_test_video = sample_test_video.permute(0, 3, 1, 2)

    # Adjust the dimension order to(num_frames,num_channels,  height, width)
    return sample_test_video

def getoptflow(on=None,apex=None,off=None,args=None):
    # use_optflow 1 2 3They are pure optical flow, respectively Optical flow corresponds to the graph One optical flow, one image
    # ("############################ use opt flow #########################")
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    cv2.setNumThreads(1)
    frame1 = cv2.imread(on)
    frame2 = cv2.imread(apex)
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # initialization Lucas-Kanade Method parameters
    lk_params = dict(winSize=(15, 15), 
                    maxLevel=2, 
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Searching for feature points in the first frame，Set here maxCorners by 100
    prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Calculate optical flow
    next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_points, None, **lk_params)

    # Select good feature points
    good_new = next_points[status == 1]
    good_prev = prev_points[status == 1]

    # Draw optical flow trajectory
    if args.use_optflow == 1:
        result = np.zeros_like(frame2)
    elif args.use_optflow == 2:
        result = np.copy(frame2)   
    elif args.use_optflow == 3:
        result = np.zeros_like(frame2)
    
        
    for i, (new, old) in enumerate(zip(good_new, good_prev)):
        a, b = new.ravel()
        c, d = old.ravel()
        result = cv2.line(result, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 5)
        result = cv2.circle(result, (int(a), int(b)), 2, (0, 0, 255), -1)
        result = cv2.circle(result,  (int(c), int(d)), 2, (0, 0, 255), -1)

    img1 = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    frame1 = cv2.imread(apex)
    frame2 = cv2.imread(off)
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # initialization Lucas-Kanade Method parameters
    lk_params = dict(winSize=(15, 15), 
                    maxLevel=2, 
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Searching for feature points in the first frame，Set here maxCorners by 100
    prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Calculate optical flow
    next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_points, None, **lk_params)

    # Select good feature points
    good_new = next_points[status == 1]
    good_prev = prev_points[status == 1]

    # Draw optical flow trajectory
    if args.use_optflow == 1:
        result = np.zeros_like(frame2)
    elif args.use_optflow == 2:
        result = np.copy(frame2)   
    elif args.use_optflow == 3:
        result = np.copy(frame2)
    for i, (new, old) in enumerate(zip(good_new, good_prev)):
        a, b = new.ravel()
        c, d = old.ravel()
        result = cv2.line(result, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 5)
        result = cv2.circle(result, (int(a), int(b)), 2, (0, 0, 255), -1)
        result = cv2.circle(result,  (int(c), int(d)), 2, (0, 0, 255), -1)

    img2 = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
    return Image.fromarray(img1),Image.fromarray(img2)

def get_depth(args):
    
    if (args.use_optflow!=0):
        print("############################ use opt flow #########################")
    if "only_csta" in args.model:
        return [[0,0,0]]
    if "only_usta" in args.model:
        return [[1,1,0]]
    if "nousta" in args.model:
        return [[1,1,0]]
    if args.depth_choice_num==0:
        if args.hsta_num!=0:
            return [[1, 4, 0] for _ in range(args.hsta_num)]
        else:
            return [[1, 4, 0], [1, 4, 0], [1, 4, 0]]
    elif args.depth_choice_num==1:
        if args.hsta_num!=0:
            return [[1, 1, 0] for _ in range(args.hsta_num)] 
        else:       
            return [[1, 1, 0], [1, 1, 0], [1, 1, 0]]
    elif args.depth_choice_num==2:
        if args.hsta_num!=0:
            return [[1, 2, 0] for _ in range(args.hsta_num)]
        else: 
            return [[1, 2, 0], [1, 2, 0], [1, 2, 0]]
    elif args.depth_choice_num==3:
        if args.hsta_num!=0:
            return [[1, 3, 0] for _ in range(args.hsta_num)]
        else:
            return [[1, 3, 0], [1, 3, 0], [1, 3, 0]]
    elif args.depth_choice_num==4:
        if args.hsta_num!=0:
            return [[1, 4, 0] for _ in range(args.hsta_num)]
        else:
            return [[1, 4, 0], [1, 4, 0], [1, 4, 0]] 
    elif args.depth_choice_num==7:
        if args.hsta_num!=0:
            return [[1, 5, 0] for _ in range(args.hsta_num)]
        else:
            return [[1, 5, 0], [1, 5, 0], [1, 5, 0]] 
    
    elif args.depth_choice_num==5:
        if args.hsta_num!=0:
            return [[2, 2, 0] for _ in range(args.hsta_num)]
        else:
            return [[2, 2, 0], [2, 2, 0], [2, 2, 0]]          
    elif args.depth_choice_num==6:
        return [[2, 1, 0], [2, 1, 0], [2, 1, 0]] 
    elif args.depth_choice_num==8:
        if args.hsta_num!=0:
            return [[3, 1, 0] for _ in range(args.hsta_num)]
        else:
            return [[3, 1, 0], [3, 1, 0], [3, 1, 0]] 
    elif args.depth_choice_num==9:
        if args.hsta_num!=0:
            return [[3, 2, 0] for _ in range(args.hsta_num)]
        else:
            return [[3, 2, 0],[3, 2, 0],[3, 2, 0]]    
    elif args.depth_choice_num==10:
        if args.hsta_num!=0:
            return [[3, 3, 0] for _ in range(args.hsta_num)]
        else:
            return [[3, 3, 0] ,[3, 3, 0] ,[3, 3, 0] ]    
    
def getsave_root(args=None):
    if(args.data_set=="smic" or args.data_set=="samm"  or args.data_set=="casme2"):
        return "YOURPATH/usta_cross_uni/usta/scripts/other_dataset_cmp/result"
    if(args.data_set=="casme3"):
        return "YOURPATH/usta_cross_uni/usta/scripts/cmp_exp_casme3/result"
    
if __name__ == '__main__':

    class Args:
        def __init__(self, model=None):
            self.model = model
            
    args = Args(model="cross_uni")
    csv=pd.read_csv("YOURPATH/casme3_data/wash_new_all/new_labels/Objective_class_as_label/use_macro_exp_train_fold_1.csv")

    print(csv.head(10))
    sample="YOURPATH/casme3_data/wash_new_all/data/spNO.179/i/color"
    buffer=new_load_video_from_path(sample=sample,index=4,df=csv,args=args) 
    print(buffer.shape)