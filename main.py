import argparse
import random
from pathlib import Path
import itertools
from os import listdir
from os.path import isfile, join
import numpy as np
import torch
import torch.nn.functional as F
import hotr.data.transforms.transforms as T
import hotr.util.misc as utils
from hotr.engine.arg_parser import get_args_parser
from hotr.models import build_model
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
ACTION_NAMES =  [
    'adjusting', 'assembling', 'blocking', 'blowing', 'boarding', 'breaking', 'brushing with', 'buying', 'carrying', 'catching', 
    'chasing', 'checking', 'cleaning', 'controling', 'cooking', 'cutting', 'cutting with', 'directing', 'draging', 'dribbling', 'drinking with', 'drive', 
    'drying', 'eating', 'eating at', 'exiting', 'feeding', 'filling', 'fliping', 'flushing', 'flying', 'greeting', 'grinding', 'grooming', 'herding', 'hitting', 
    'holding', 'hopping on', 'hosing', 'huging', 'huntting', 'inspectting', 'installing', 'jumping', 'kicking', 'kissing', 'lassoing', 'launching', 'licking', 
    'lying on', 'lifting', 'lighting', 'loading', 'losing', 'making', 'milking', 'moving', 'no_interaction', 'opening', 'operating', 'packing', 
    'painting', 'parking', 'paying', 'peeling', 'petting', 'picking', 'picking up', 'pointing', 'pouring', 'pulling', 'pushing', 'racing', 'reading', 'releasing', 
    'repairing', 'riding', 'rowing', 'running', 'sailing', 'scratching', 'serving', 'seting', 'shearing', 'signing', 'siping', 'sitting at', 'sitting on', 
    'sliding', 'smelling', 'spining', 'squeezing', 'stabing', 'standing on', 'standing under', 'sticking', 'stiring', 'stopping at', 'straddling', 
    'swinging', 'tagging', 'talking on', 'teaching', 'texting on', 'throwing', 'tying', 'toasting', 'training', 'turning', 'typing on', 'walking', 'washing', 
    'watching', 'waving', 'wearing', 'wielding', 'zipping']
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]          
CLASS_COLORS = [
    ( 255, 0, 0),( 0, 255, 0),( 0, 0, 255),( 255, 255, 0),( 255, 165, 0),( 128, 0, 128),( 255, 192, 203),
    ( 0, 0, 0),( 255, 255, 255),( 128, 128, 128),( 0, 255, 255),( 255, 0, 255),( 127, 255, 0),( 0, 128, 128),( 128, 0, 0),
    ( 0, 0, 128),( 230, 230, 250),( 64, 224, 208),( 255, 215, 0),( 192, 192, 192),( 128, 128, 0),( 75, 0, 130),
    ( 245, 245, 220),( 255, 0, 255),( 255, 218, 185),( 250, 128, 114),( 160, 82, 45),( 135, 206, 235),( 106, 90, 205),
    ( 210, 180, 140),( 255, 99, 71),( 238, 130, 238),( 245, 222, 179),( 240, 230, 140),( 255, 255, 240),( 189, 252, 201),
    ( 70, 130, 180),( 139, 0, 0),( 0, 100, 0),( 255, 127, 80),( 220, 20, 60),( 0, 0, 139),( 169, 169, 169),
    ( 255, 20, 147),( 34, 139, 34),( 218, 165, 32),( 255, 105, 180),( 205, 92, 92),( 255, 250, 205),( 173, 216, 230),
    ( 144, 238, 144),( 211, 211, 211),( 255, 255, 224),( 0, 255, 0),( 224, 176, 255),( 0, 0, 205),( 128, 128, 128),
    ( 0, 0, 128),( 152, 251, 152),( 204, 204, 255),( 221, 160, 221),( 176, 224, 230),( 188, 143, 143),( 65, 105, 225),
    ( 139, 69, 19),( 244, 164, 96),( 46, 139, 87),( 112, 128, 144),( 255, 250, 250),( 0, 255, 127),( 216, 191, 216),( 154, 205, 50),
    ( 250, 235, 215),( 240, 255, 255),( 255, 228, 196),( 255, 235, 205),( 222, 184, 135),( 95, 158, 160),( 210, 105, 30),
    ( 100, 149, 237),( 0, 139, 139)
]

INTERACTION_COLORS = [
    ( 184, 134, 11),( 189, 183, 107),( 139, 0, 139),( 85, 107, 47),( 255, 140, 0),( 153, 50, 204),( 233, 150, 122),( 72, 61, 139),
    ( 0, 206, 209),( 0, 191, 255),( 30, 144, 255),( 178, 34, 34),
]

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1).to('cuda')

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, final_triplets):
  #bbox_persons, objs, bbox_objs, actions
    subs = (t[0] for t in final_triplets)
    bbox_subs = (t[1] for t in final_triplets)
    objs = (t[2] for t in final_triplets)
    bbox_objs = (t[3] for t in final_triplets)
    actions = (t[4] for t in final_triplets)
    scores = (t[5] for t in final_triplets)

    plt.figure(figsize=(16,10))
    plt.axis('off')
    plt.imshow(pil_img)
    ax = plt.gca()
        
    class_colors = []
    for c in CLASS_COLORS:
      c = tuple(ci/255 for ci in c)
      class_colors.append(c)

    interaction_colors = []
    for c in INTERACTION_COLORS:
      c = tuple(ci/255 for ci in c)
      interaction_colors.append(c)

    text_pos = []
    for sub, (xsmin, ysmin, xsmax, ysmax), obj, (xomin, yomin, xomax, yomax), action in zip(subs, bbox_subs, objs, bbox_objs, actions):
        if ACTION_NAMES[action] in ['no_interaction', 'walk_agent', 'smile_agent', 'run_agent', 'stand_agent']:
          continue
        ax.add_patch(plt.Rectangle((xsmin, ysmin), xsmax - xsmin, ysmax - ysmin,
                                   fill=False, color=class_colors[0], linewidth=3))
        ax.add_patch(plt.Rectangle((xomin, yomin), xomax - xomin, yomax - yomin,
                                   fill=False, color=class_colors[obj], linewidth=3))
        xcs, ycs = xsmin + (xsmax - xsmin) / 2 , ysmin + (ysmax - ysmin) / 2
        xco, yco = xomin + (xomax - xomin) / 2 , yomin + (yomax - yomin) / 2
        s = [xcs, ycs]
        o = [xco, yco]
        plt.plot([xcs,xco],[ycs,yco], linewidth=3, color = class_colors[obj], marker = 'o')
        text_obj = f'{CLASSES[obj]}'
        text_interaction = f'{ACTION_NAMES[action]}'
        if obj != 0:
          ax.text(xomin, yomin, text_obj, fontsize=15,
                bbox=dict(facecolor=class_colors[obj], alpha=0.5))
        while [xsmin,ysmin] in text_pos:
          ysmin -= 20
        text_pos.append([xsmin,ysmin])
        ax.text(xsmin, ysmin, text_interaction, fontsize=15, color = 'white', bbox=dict(facecolor=interaction_colors[action%12],alpha=0.5))
    plt.savefig('./Visualize/'+ "result.png" )
    plt.close()

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main(args):
    utils.init_distributed_mode(args)

    if args.frozen_weights is not None:
        print("Freeze weights for detector")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Data setup
    args.num_classes = 91
    args.num_actions = 117
    args.action_names = [
      'adjust', 'assemble', 'block', 'blow', 'board', 'break', 'brush_with', 
      'buy', 'carry', 'catch', 'chase', 'check', 'clean', 'control', 'cook', 
      'cut', 'cut_with', 'direct', 'drag', 'dribble', 'drink_with', 'drive', 
      'dry', 'eat', 'eat_at', 'exit', 'feed', 'fill', 'flip', 'flush', 'fly', 
      'greet', 'grind', 'groom', 'herd', 'hit', 'hold', 'hop_on', 'hose', 
      'hug', 'hunt', 'inspect', 'install', 'jump', 'kick', 'kiss', 'lasso', 
      'launch', 'lick', 'lie_on', 'lift', 'light', 'load', 'lose', 'make', 
      'milk', 'move', 'no_interaction', 'open', 'operate', 'pack', 'paint', 
      'park', 'pay', 'peel', 'pet', 'pick', 'pick_up', 'point', 'pour', 'pull', 
      'push', 'race', 'read', 'release', 'repair', 'ride', 'row', 'run', 'sail', 
      'scratch', 'serve', 'set', 'shear', 'sign', 'sip', 'sit_at', 'sit_on', 
      'slide', 'smell', 'spin', 'squeeze', 'stab', 'stand_on', 'stand_under', 
      'stick', 'stir', 'stop_at', 'straddle', 'swing', 'tag', 'talk_on', 
      'teach', 'text_on', 'throw', 'tie', 'toast', 'train', 'turn', 'type_on', 
      'walk', 'wash', 'watch', 'wave', 'wear', 'wield', 'zip']
    if args.share_enc: args.hoi_enc_layers = args.enc_layers
    if args.pretrained_dec: args.hoi_dec_layers = args.dec_layers

    args.valid_obj_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    # Model Setup
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    #n_parameters = print_params(model)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # Weight Setup
    if args.frozen_weights is not None:
        if args.frozen_weights.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.frozen_weights, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    if args.eval:
        # test only mode

        image_path = args.input_path
        im = Image.open(image_path, 'r')
        
        try:
            img = transform(im).unsqueeze(0).to('cuda')
        except:
            raise ValueError("Error processing the image.")
        preds = []
        output_preds = []
    
        model.eval()
        outputs = model(img)
    
        orig_size = torch.tensor([torch.tensor(im.size[1]), torch.tensor(im.size[0])])
        size = torch.tensor([torch.tensor(im.size[1]), torch.tensor(im.size[0])])

        targets = [{'orig_size': orig_size, 'size': size}]

        target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to('cuda')
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        
        out_obj_logits, out_verb_logits = outputs['pred_obj_logits'], outputs['pred_actions']
        # actions
        matching_scores = (1-out_verb_logits.sigmoid()[..., -1:]) #* (1-out_verb_logits.sigmoid()[..., 57:58])
        verb_scores = out_verb_logits.sigmoid()[..., :-1] * matching_scores

        # hbox, obox
        outputs_hrepr, outputs_orepr = outputs['pred_hidx'], outputs['pred_oidx']
        obj_scores, obj_labels = F.softmax(out_obj_logits, -1)[..., :-1].max(-1)

        h_prob = F.softmax(outputs_hrepr, -1)
        h_idx_score, h_indices = h_prob.max(-1)

        # targets
        o_prob = F.softmax(outputs_orepr, -1)
        o_idx_score, o_indices = o_prob.max(-1)

        # hidx, oidx
        sub_boxes, obj_boxes = [], []
        for batch_id, (box, h_idx, o_idx) in enumerate(zip(boxes, h_indices, o_indices)):
            sub_boxes.append(box[h_idx, :])
            obj_boxes.append(box[o_idx, :])
        sub_boxes = torch.stack(sub_boxes, dim=0)
        obj_boxes = torch.stack(obj_boxes, dim=0)
    
        # accumulate results (iterate through interaction queries)
        results = []
        for os, ol, vs, ms, sb, ob in zip(obj_scores, obj_labels, verb_scores, matching_scores, sub_boxes, obj_boxes):
            sl = torch.full_like(ol, 0) # self.subject_category_id = 0 in HICO-DET
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})
            vs = vs * os.unsqueeze(1)
            ids = torch.arange(b.shape[0])
            res_dict = {
                'verb_scores': vs.to('cpu'),
                'sub_ids': ids[:ids.shape[0] // 2],
                'obj_ids': ids[ids.shape[0] // 2:],
                #'hoi_recognition_time': hoi_recognition_time
            }
            results[-1].update(res_dict)

        max_hois = 100
        #correct_mat = np.load("../hico_20160224_det/annotations/corre_hico.npy")
    
    
        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        for img_preds in preds:
            img_preds = {k: v.to('cpu').detach().numpy() for k, v in img_preds.items() if k != 'hoi_recognition_time'}
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_preds['boxes'], img_preds['labels'])]
            hoi_scores = img_preds['verb_scores']
            verb_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
            subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
            object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

            hoi_scores = hoi_scores.ravel()
            verb_labels = verb_labels.ravel()
            subject_ids = subject_ids.ravel()
            object_ids = object_ids.ravel()

            if len(subject_ids) > 0:
               object_labels = np.array([bboxes[object_id]['category_id'] for object_id in object_ids])
               #masks = correct_mat[verb_labels, object_labels]
               hoi_scores *= 10
            
               hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                       subject_id, object_id, category_id, score in zip(subject_ids, object_ids, verb_labels, hoi_scores)]
               hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
               hois = hois[:max_hois]
            else:
               hois = []
            
            output_preds.append({
                'predictions': bboxes,
                'hoi_prediction': hois
            })
        triplets = []
        for img_preds in output_preds:
            pred_bboxes = img_preds['predictions']
            pred_hois = img_preds['hoi_prediction']
            pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
            if len(pred_hois) != 0:
                for pred_hoi in pred_hois:
                    triplets.append([pred_bboxes[pred_hoi['subject_id']]['category_id'], pred_bboxes[pred_hoi['subject_id']]['bbox'].tolist(),  pred_bboxes[pred_hoi['object_id']]['category_id'], pred_bboxes[pred_hoi['object_id']]['bbox'].tolist(),
                            pred_hoi['category_id'], pred_hoi['score']])
        final_triplets = []
        sub_obj = []
        for triplet in triplets:
            if triplet[-1] < 0.9:
                continue
            if [triplet[0], triplet[1], triplet[2], triplet[3], triplet[4]] not in sub_obj:
                final_triplets.append(triplet)
                sub_obj.append([triplet[0], triplet[1], triplet[2], triplet[3], triplet[4]])
        #import pdb; pdb.set_trace()
        if len(final_triplets) != 0:
            plot_results(im, final_triplets)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'End-to-End Human Object Interaction training and evaluation script',
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir += f"/{args.group_name}/{args.run_name}/"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)