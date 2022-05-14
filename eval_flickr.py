import torch
import torchvision
import os,sys
import numpy as np
from pathlib import Path
import pycocotools
import torch.utils
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler
from torch import nn
import torch.nn.functional as F
from functools import partial
import json
from transformers import RobertaTokenizerFast
import pickle
import time
from collections import defaultdict
import argparse
import pandas as pd
import wandb

class PostProcessFlickr(nn.Module):
    """This module converts the model's output for Flickr30k entities evaluation.
    This processor is intended for recall@k evaluation with respect to each phrase in the sentence.
    It requires a description of each phrase (as a binary mask), and returns a sorted list of boxes for each phrase.
    """

    @torch.no_grad()
    def forward(self, outputs, target_sizes, positive_map, items_per_batch_element):
        """Perform the computation.
        Args:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            positive_map: tensor [total_nbr_phrases x max_seq_len] for each phrase in the batch, contains a binary
                          mask of the tokens that correspond to that sentence. Note that this is a "collapsed" batch,
                          meaning that all the phrases of all the batch elements are stored sequentially.
            items_per_batch_element: list[int] number of phrases corresponding to each batch element.
            captions : list of captions for all elements in batch
            mask_token_idx : list of len(batch_size) where each element indicates the index of the positive_token_eval 
                             that is masked out/ replaced
        """
        from util import box_ops
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        batch_size = target_sizes.shape[0]

        prob = F.softmax(out_logits, -1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # and from relative [0, 1] to absolute [0, height] coordinates
        boxes = boxes * scale_fct[:, None, :]
        cum_sum = np.cumsum(items_per_batch_element)

        curr_batch_index = 0
        # binarize the map if not already binary
        pos = positive_map > 1e-6
      
        predicted_boxes = [[] for _ in range(batch_size)]
        scores_output = [[] for _ in range(batch_size)]
        pos_tokens_pred = [[] for _ in range(batch_size)]

        # The collapsed batch dimension must match the number of items
        assert len(pos) == cum_sum[-1]

        if len(pos) == 0:
            return predicted_boxes

        # if the first batch elements don't contain elements, skip them.
        while cum_sum[curr_batch_index] == 0:
            curr_batch_index += 1
        phrase_ids = [list(range(i)) for i in items_per_batch_element]
        for i in range(len(pos)):

            # scores are computed by taking the max over the scores assigned to the positive tokens
            scores, _ = torch.max(pos[i].unsqueeze(0) * prob[curr_batch_index, :, :], dim=-1)
            _, indices = torch.sort(scores, descending=True)
            assert items_per_batch_element[curr_batch_index] > 0
            predicted_boxes[curr_batch_index].append(boxes[curr_batch_index][indices].to("cpu").tolist())
            scores_output[curr_batch_index].append(scores[indices].to("cpu").tolist())
            assert len(predicted_boxes[curr_batch_index]) == len(scores_output[curr_batch_index]), f"len(predicted_boxes[curr_batch_index]): {len(predicted_boxes[curr_batch_index])} and len(scores_output[curr_batch_index]): {len(scores_output[curr_batch_index])}"
            if i == len(pos) - 1:
                break

            # check if we need to move to the next batch element
            while i >= cum_sum[curr_batch_index] - 1:
                curr_batch_index += 1
                assert curr_batch_index < len(cum_sum)
        
        return predicted_boxes, phrase_ids, scores_output

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img_dir",
        default=None,
        type=str,
        required=True,
        help="The image data directory.",
    )

    parser.add_argument(
        "--annotations_dir",
        default=None,
        type=str,
        required=False,
        help="Directory where annotations file is [DEPRECATED].",
    )

    parser.add_argument(
        "--mdetr_git_dir",
        default=None,
        type=str,
        required=True,
        help="Git directory for loading MDETR-specific functions.",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--batch_size",
        default = 10,
        type= int,
        required = True,
        help= "batch size for "
    )

    parser.add_argument(
        "--pretrained_model",
        default = 'mdetr_resnet101',
        type = str,
        required = True,
        choices = ['mdetr_efficientnetB5', 'mdetr_efficientnetB3', 'mdetr_resnet101'],
        help = "name of pretrained MDETR model to load"
    )

    parser.add_argument(
        "--gpu_type",
        default = 'cpu',
        type = str,
        required = True,
        choices = ['cpu', 'v100', 'rtx8000', 'p100'],
        help = "Type of GPU being used for inference"
    )

    args = parser.parse_args()

    job_name = f"eval-{args.pretrained_model}-bs{args.batch_size}-{args.gpu_type}"

    run = wandb.init(project="text-od-robustness", 
                entity="vector-victors",
                config=vars(args),
                name=job_name,
                job_type="eval")

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    sys.path.append(args.mdetr_git_dir)
    import datasets.transforms as T
    from datasets.coco import ModulatedDetection, CocoDetection, convert_coco_poly_to_mask, create_positive_map, ConvertCocoPolysToMask, make_coco_transforms
    from datasets.phrasecut_utils import data_transfer
    import util.misc as utils
    from util.misc import targets_to
    from util import box_ops
    from models.mdetr import MDETR
    from datasets.refexp import RefExpDetection
    from util.metrics import MetricLogger
    from datasets import get_coco_api_from_dataset
    from datasets.flickr_eval import FlickrEvaluator

    with open('data/flickr_test_masked.json') as f:
        flickr_anns = json.load(f)

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print("Using", device)
    #from models.postprocessors import build_postprocessors
    def build_dataset(img_dir, ann_file, image_set, text_encoder_type):
        tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
        dataset = RefExpDetection(
            img_dir,
            ann_file,
            transforms=make_coco_transforms(image_set, cautious=True),
            return_masks=False,
            return_tokens=True,
            tokenizer=tokenizer,
        )
        return dataset

    #Create coco-formatted dataset
    test_dset = build_dataset(img_dir = args.img_dir, 
                              ann_file = 'data/flickr_test_masked.json', 
                              image_set = 'val', 
                              text_encoder_type= "roberta-base")

    num_workers = 8
    print(f"Creating test dataset with {len(test_dset)} instances, using {num_workers} workers")

     #Set up dataloader
    test_loader = DataLoader(test_dset,
                            args.batch_size,
                            sampler = torch.utils.data.SequentialSampler(test_dset),
                            drop_last = False,
                            num_workers=num_workers,
                            collate_fn=partial(utils.collate_fn, False)
                            )

    start = time.time()

    #Load pretrained models
    model, postprocessor = torch.hub.load('ashkamath/mdetr:main', args.pretrained_model, pretrained=True, return_postprocessor=True)
    model = model.to(device)
    end = time.time()
    print(f"Loaded {args.pretrained_model} in {end-start} seconds", flush = True)
    with torch.no_grad():
        model.eval()

    base_ds = get_coco_api_from_dataset(test_dset)

    #Set up metric logger object from MDETR repo
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    start_time = time.time()

    #Eval Loop
    i = 0
    flickr_res_collector = []
    mask_res_collector = []
    time_df= pd.DataFrame()

    with torch.no_grad():
        #start time for full eval loop
        start_time = time.time()
        for batch_dict in metric_logger.log_every(test_loader, args.batch_size, header):
            #start time for full batch processing
            batch_start_time = time.time()

            #Extract data, targets, positive token map & captions from targets in dataloader
            samples = batch_dict['samples'].to(device)
            positive_map = batch_dict["positive_map"].to(device)
            targets = batch_dict["targets"]
            captions = [t["caption"] for t in targets]
            targets = targets_to(targets, device)

            #Get size of original image - this is used in evaluation when we're scaling boxes to image sizes 
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            orig_target_sizes = orig_target_sizes.to(device)
            
            #Get start time for scoring/ model outputs
            scoring_start_time = time.time()
            memory_cache = model(samples, captions, encode_and_save=True)
            output = model(samples, captions,encode_and_save=False, memory_cache=memory_cache)
            scoring_end_time = time.time()

            #Get start time for post-processing
            post_process_start_time = time.time()

            #Extract metadata that is necessary to pass to postprocessor
            image_ids = [t["image_id"] for t in targets]   #unique image id
            sentence_ids = [t["sentence_id"] for t in targets]    ## caption for a given image
            items_per_batch_element = [t["nb_eval"] for t in targets]   #Each image has a certain numnber of phrases/ objects. This is the number (get from postiive_tokens_eval in dataloader)
            
            #return list of indices for the masked positive token eval map
            image_anns = [image for image in flickr_anns['images'] if image['id'] in(image_ids)]
            #assert len(image_anns)==args.batch_size

            #get the index of the masked token in the tokens_positive_eval
            mask_token_idx = [image['tokens_positive_eval_idx'] for image in image_anns]

            #positive_map_eval is a binary map fo tokens for each sentence. Gets passed to post-processor
            positive_map_eval = batch_dict["positive_map_eval"].to(device)

            bboxes, phrase_ids, scores = PostProcessFlickr()(output, orig_target_sizes, positive_map_eval, 
                                                            items_per_batch_element)
            
            post_process_end_time = time.time()
            flickr_res = []
            mask_res = []
            
            #Save outputs from postprocessor in a list
            for im_id, sent_id, boxes, phrase_id, score, mask_idx in zip(image_ids, sentence_ids, bboxes, phrase_ids, scores, mask_token_idx):
                flickr_res.append({"image_id": im_id, "sentence_id":sent_id,
                                    "boxes": boxes,
                                    "phrase_ids": phrase_id, 
                                    "scores": score})
            
                mask_res.append({"image_id": im_id, "sentence_id":sent_id,
                                "boxes": boxes[mask_idx],
                                "scores": score[mask_idx]})
            post_process_end_time = time.time()
            batch_end_time = time.time()
            
            if i% 100 == 0:
                print(f"FINISHING BATCH {i}")
                print(f"batch processing time:{batch_end_time-batch_start_time}")
                print(f"total processing time:{batch_end_time-start_time}")
                print("---------------------------------------------")
                print("")

            for f in mask_res:
                mask_res_collector.append(f)
            
            pkl_file = open(os.path.join(args.output_dir, f'{args.gpu_type}_{args.pretrained_model}_{args.batch_size}_masked_token_results.pkl'), 'wb')
            pickle.dump(mask_res_collector, pkl_file)
            pkl_file.close()

            with open(os.path.join(args.output_dir, f'{args.gpu_type}_{args.pretrained_model}_{args.batch_size}_flickr_results.pkl'), 'wb') as f:
                pickle.dump(flickr_res, f)

            add_time = {'total_cumulative_time': time.time()-start_time,
                        'batch_time': batch_end_time - batch_start_time,
                        'model_scoring_time': scoring_end_time - scoring_start_time,
                        'post_processing_time': post_process_end_time - post_process_start_time}
            

            time_df = time_df.append(add_time, ignore_index = True)
            time_df.to_csv(os.path.join(args.output_dir, f'{args.gpu_type}_{args.pretrained_model}_{args.batch_size}_eval_time.csv'))
            i+=1



if __name__ == "__main__":
    main()