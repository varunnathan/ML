import os, argparse


# GLOBALS
parser = argparse.ArgumentParser()
parser.add_argument('--segments', type=str,
                    help="comma separated segment lst")
parser.add_argument('task', choices=["prepare-data", "feature-calc"],
                    help="task to perform")
parser.add_argument('--file-num', type=int, help='file number of user data')
parser.add_argument('--feature-type', choices=['baseline', 'neighbourhood',
                                               'surpriseMF', 'lightFM'],
                    default='baseline', help='feature type')
parser.add_argument('--data-type', choices=['user_data', 'qualifying_data',
                                          'segment_data'], default='user_data',
                    help='data type for preparation')
parser.add_argument('--feature-task', choices=['training', 'validation', 'both'],
                    default='both', help='feature calculation task')
args = parser.parse_args()
TASK = args.task
FILE_NUM = args.file_num
SEGMENTS = args.segments
FEATURE_TYPE = args.feature_type
DATA_TYPE = args.data_type
FEATURE_TASK = args.feature_task

def main(task, segment):
    if task == 'prepare-data':
        print('user data preparation')
        os.system('python app.py prepare-data --file-num {} --data-type {}'.format(
            FILE_NUM, 'user_data'))
    elif task == 'feature-calc':
        print('feature calculation')
        os.system('python app.py feature-calc --file-num {} --segment {} --feature-type {} --feature-task {}'.format(
            FILE_NUM, segment, FEATURE_TYPE, FEATURE_TASK))


if __name__ == '__main__':
    print('execution begins...\n')
    segments = [str(x) for x in SEGMENTS.split(',')]

    for segment in segments:
        print('Segment: ', segment)
        main(TASK, segment)
