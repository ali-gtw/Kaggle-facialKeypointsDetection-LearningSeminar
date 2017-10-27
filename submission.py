import pandas as pd


mappings = {
      'left_eye_center_x':          0,
      'left_eye_center_y':          1,
      'right_eye_center_x':         2,
      'right_eye_center_y':         3,
      'left_eye_inner_corner_x':    4,
      'left_eye_inner_corner_y':    5,
      'left_eye_outer_corner_x':    6,
      'left_eye_outer_corner_y':    7,
      'right_eye_inner_corner_x':   8,
      'right_eye_inner_corner_y':   9,
      'right_eye_outer_corner_x':   10,
      'right_eye_outer_corner_y':   11,
      'left_eyebrow_inner_end_x':   12,
      'left_eyebrow_inner_end_y':   13,
      'left_eyebrow_outer_end_x':   14,
      'left_eyebrow_outer_end_y':   15,
      'right_eyebrow_inner_end_x':  16,
      'right_eyebrow_inner_end_y':  17,
      'right_eyebrow_outer_end_x':  18,
      'right_eyebrow_outer_end_y':  19,
      'nose_tip_x':                 20,
      'nose_tip_y':                 21,
      'mouth_left_corner_x':        22,
      'mouth_left_corner_y':        23,
      'mouth_right_corner_x':       24,
      'mouth_right_corner_y':       25,
      'mouth_center_top_lip_x':     26,
      'mouth_center_top_lip_y':     27,
      'mouth_center_bottom_lip_x':  28,
      'mouth_center_bottom_lip_y':  29
    }






def submit (net, load) :

    X, _ = load(test=True)
    y_pred = net.predict(X)
    y_pred = y_pred 

    

    outputs = []
    base_df = pd.read_csv('data/IdLookupTable.csv')

    for imageId in range (1, y_pred.shape[0]+1):
        prediction = y_pred[imageId-1]
        specified_image_id_df = base_df[base_df['ImageId'] == imageId]
        for _, row in specified_image_id_df.iterrows():
            row_id = row['RowId']
            feature_name = row['FeatureName']
            mapped_index = mappings[feature_name]
            pred_value = prediction[mapped_index] * 48 + 48
            if pred_value > 96.0:
                pred_value = 96.0
            if pred_value < 0.0:
                pred_value = 0.0
            outputs.append([row_id, pred_value])


    output_df = pd.DataFrame(outputs, columns=['RowId', 'Location'])
    output_df.to_csv("output.csv", index=False)
