import argparse
import os
from models import Dataset
from os.path import join
import pandas as pd
from random import shuffle, random


def create_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Dataset partitioning for the dataset collector framework'
        )
    )

    parser.add_argument(
        '-d',
        '--data_path',
        help='path to dataset folder ',
        default=os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    parser.add_argument(
        "--random",
        action="store_true")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    d = Dataset(args.data_path)
    d.initialize_from_filesystem()
    
    for acquisition in d.acquisitions:
        output = [[] for _ in range(6)] 
        split =  [[] for _ in range(6)] 

        for image in acquisition.images:

            obstacle_visible = image.labels['l_p_of_collision'] or image.labels['c_p_of_collision'] or image.labels['r_p_of_collision'] 
            if image.labels["edge"] == 'E_label.NO_EDGE' and obstacle_visible == True:
                output[0].append(image.name)
            elif image.labels["edge"] == 'E_label.EDGE'    and obstacle_visible == True:
                output[1].append(image.name)
            elif image.labels["edge"] == 'E_label.CORNER'  and obstacle_visible == True:
                output[2].append(image.name)
            elif image.labels["edge"] == 'E_label.NO_EDGE' and obstacle_visible == False:
                output[3].append(image.name)
            elif image.labels["edge"] == 'E_label.EDGE'    and obstacle_visible == False:
                output[4].append(image.name)
            elif image.labels["edge"] == 'E_label.CORNER' and obstacle_visible == False:
                output[5].append(image.name)
            else:
                raise ValueError

        if args.random:
            for sample in output:
                shuffle(sample)



        # division in training (70%),  validation (10%) and testing (20%) for each acquisition 
        for i in range(len(output)):
            for j in range(len(output[i])):
                if   j < 0.7*(len(output[i])-1): split[i].append("train")
                elif j > 0.8*(len(output[i])-1): 
                    if (i == 0) or (i == 5):
                        split[i].append("test")
                    else:
                        split[i].append("test")
                else: split[i].append("valid")

        old_df_path = join(acquisition.path, acquisition.LABELS_FILENAME)      
        df2 = pd.read_csv(old_df_path)
        df2.insert(1, "partition", None)


        # # Create new label dictionary 
        final = {"filename": [], "partition": [], "labels": []}
        for row_idx, row in df2.iterrows():
            for i in range(len(output)):
                if row['filename'] in output[i]:
                    df2['partition'][row_idx] = split[i][output[i].index(row['filename'])]

        df2.to_csv(join(acquisition.path,"labels_partitioned.csv"), index=False)

if __name__ == "__main__":
    main()
