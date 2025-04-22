import heapq
import json
import numpy as np
import argparse
import os
from decord import VideoReader
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract Video Feature')

    # video and json path
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--score_path', type=str, required=True, help='Path to the JSON file containing frame scores')
    parser.add_argument('--frame_path', type=str, required=True, help='Path to the JSON file containing frame names')

    # DO NOT CHANGE!
    parser.add_argument('--max_num_frames', type=int, default=5, help='Maximum number of frames to select')
    parser.add_argument('--ratio', type=int, default=1, help='Frame ratio for down-sampling')
    parser.add_argument('--t1', type=int, default=0.8, help='Threshold t1 for splitting frames')
    parser.add_argument('--t2', type=int, default=-100, help='Threshold t2 for splitting frames')
    parser.add_argument('--all_depth', type=int, default=2, help='Maximum depth for frame splitting')
    parser.add_argument('--output_file', type=str, default='./selected_frames', help='Output path for selected frames')
    
    return parser.parse_args()

def meanstd(len_scores, dic_scores, n, fns,t1,t2,all_depth):
        split_scores = []
        split_fn = []
        no_split_scores = []
        no_split_fn = []
        i= 0
        for dic_score, fn in zip(dic_scores, fns):
                # normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))
                score = dic_score['score']
                depth = dic_score['depth']
                mean = np.mean(score)
                std = np.std(score)

                top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
                top_score = [score[t] for t in top_n]
                # print(f"split {i}: ",len(score))
                i += 1
                mean_diff = np.mean(top_score) - mean
                if mean_diff > t1 and std > t2:
                        no_split_scores.append(dic_score)
                        no_split_fn.append(fn)
                elif depth < all_depth:
                # elif len(score)>(len_scores/n)*2 and len(score) >= 8:
                        score1 = score[:len(score)//2]
                        score2 = score[len(score)//2:]
                        fn1 = fn[:len(score)//2]
                        fn2 = fn[len(score)//2:]                       
                        split_scores.append(dict(score=score1,depth=depth+1))
                        split_scores.append(dict(score=score2,depth=depth+1))
                        split_fn.append(fn1)
                        split_fn.append(fn2)
                else:
                        no_split_scores.append(dic_score)
                        no_split_fn.append(fn)
        if len(split_scores) > 0:
                all_split_score, all_split_fn = meanstd(len_scores, split_scores, n, split_fn,t1,t2,all_depth)
        else:
                all_split_score = []
                all_split_fn = []
        all_split_score = no_split_scores + all_split_score
        all_split_fn = no_split_fn + all_split_fn

        return all_split_score, all_split_fn


def main(args):
    print(333)
    max_num_frames = args.max_num_frames
    ratio = args.ratio
    t1 = args.t1
    t2 = args.t2
    all_depth = args.all_depth
    outs = []
    segs = []
    
    if not os.path.exists(args.output_file):
        os.makedirs(args.output_file)

    with open(args.score_path) as f:
        itm_outs = json.load(f)
    with open(args.frame_path) as f:
        fn_outs = json.load(f)

    for itm_out,fn_out in zip(itm_outs,fn_outs):
        nums = int(len(itm_out)/ratio)
        new_score = [itm_out[num*ratio] for num in range(nums)]
        new_fnum = [fn_out[num*ratio] for num in range(nums)]
        score = new_score
        fn = new_fnum
        num = max_num_frames
        if len(score) >= num:
            normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))
            a, b = meanstd(len(score), [dict(score=normalized_data,depth=0)], num, [fn], t1, t2, all_depth)
            segs.append(len(a))
            out = []
            print(len(score))
            if len(score) >= num:
                print(222)
                for s,f in zip(a,b): 
                    print(111)
                    print(s, f)
                    f_num = int(num / 2**(s['depth']))
                    topk = heapq.nlargest(f_num, range(len(s['score'])), s['score'].__getitem__)
                    f_nums = [f[t] for t in topk]
                    out.extend(f_nums)
            out.sort()
            outs.append(out)
        else:
            outs.append(fn)

    out_score_path = os.path.join(args.output_file, 'selected_frames.json')
    with open(out_score_path, 'w') as f:
        json.dump(outs, f)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
