import os
import json
import re
import numpy as np
from tqdm import tqdm
import sys
import copy
import random

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from eval_webnlg.pycocotools.coco import COCO
from eval_webnlg.pycocoevalcap.eval import COCOEvalCap


def get_t_emb_dim(args):
    t_emb_dim = int(args.entity_entity)+int(args.entity_relation)\
        +int(args.relation_entity)+int(args.relation_relation)+1
    return t_emb_dim
        
        
def run_coco_eval(data_ref, data_sys):
    """Run the COCO evaluator, return the resulting evaluation object (contains both
    system- and segment-level scores."""
    # convert references and system outputs to MS-COCO format in-memory
    coco_ref = create_coco_refs(data_ref)
    coco_sys = create_coco_sys(data_sys)

    print('Running MS-COCO evaluator...', file=sys.stderr)
    coco = COCO()
    coco.dataset = coco_ref
    coco.createIndex()

    coco_res = coco.loadRes(resData=coco_sys)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()

    return coco_eval


def create_coco_refs(data_ref):
    """Create MS-COCO human references JSON."""
    out = {'info': {}, 'licenses': [], 'images': [], 'type': 'captions', 'annotations': []}
    ref_id = 0
    for inst_id, refs in enumerate(data_ref):
        out['images'].append({'id': 'inst-%d' % inst_id})
        for ref in refs:
            out['annotations'].append({'image_id': 'inst-%d' % inst_id,
                                       'id': ref_id,
                                       'caption': ref})
            ref_id += 1
    return out


def create_coco_sys(data_sys):
    """Create MS-COCO system outputs JSON."""
    out = []
    for inst_id, inst in enumerate(data_sys):
        out.append({'image_id': 'inst-%d' % inst_id, 'caption': inst})
    return out


class GAPDataloader(DataLoader):

    def __init__(self, args, dataset, mode):
        if mode == "train":
            sampler = RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(GAPDataloader, self).__init__(dataset, sampler=sampler, batch_size=batch_size,
                                               num_workers=args.num_workers)

        
        
        
        
        
        
# Downstream dataset
class WebNLGDataset(Dataset):
    def __init__(self, logger, args, data_path, tokenizer, mode):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.topology = {"entity-entity": args.entity_entity, 
                           "entity-relation": args.entity_relation,
                           "relation-entity": args.relation_entity,
                           "relation-relation": args.relation_relation
                          }  
        
        with open(self.data_path + '.json', 'r') as f:
            self.data = json.load(f)

        print("Total samples = {}".format(len(self.data)))

        if args.debug:
            self.data = self.data[:1000]
        assert type(self.data) == list
        assert all(["id" in d for d in self.data]), self.data[0].keys()
        if type(self.data[0]["id"]) == int:
            for i in range(len(self.data)):
                self.data[i]["id"] = str(self.data[i]["id"])

        self.args = args
        self.data_type = mode
        self.metric = "BLEU"

        self.head_ids, self.rel_ids, self.tail_ids = self.tokenizer.encode(' [head]', add_special_tokens=False), \
                                                     self.tokenizer.encode(' [relation]', add_special_tokens=False), \
                                                     self.tokenizer.encode(' [tail]', add_special_tokens=False)

        self.graph_ids, self.text_ids = self.tokenizer.encode(' [graph]', add_special_tokens=False), \
                                        self.tokenizer.encode(' [text]', add_special_tokens=False)

        if self.args.model_name == "bart":
            self.mask_token = self.tokenizer.mask_token
            self.mask_token_id = self.tokenizer.mask_token_id
        else:
            self.mask_token = self.tokenizer.additional_special_tokens[0]
            self.mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.additional_special_tokens[0])

        if self.args.model_name == "bart":
            if self.args.append_another_bos:
                self.add_bos_id = [self.tokenizer.bos_token_id] * 2
            else:
                self.add_bos_id = [self.tokenizer.bos_token_id]
        else:
            self.add_bos_id = []

    def __len__(self):
        return len(self.data)

    def linearize_v2(self, entity, entity_change, head_ids, rel_ids, tail_ids,
                        relation_change, cnt_edge, adj_matrix):
        # string_label: encoder ids
        # string_label_tokens: encoder tokens

        if len(entity[0]) == 0:
            return [], '', [], [], cnt_edge, adj_matrix
        nodes, edges = [], []
        string_label = copy.deepcopy(head_ids)
        string_label_tokens = ' [head]'
        nodes.extend([-1] * len(string_label))
        edges.extend([-1] * len(string_label))


        string_label += entity_change[entity[0]][0]
        string_label_tokens += ' {}'.format(entity[0])
        nodes.extend([entity_change[entity[0]][1]] * len(entity_change[entity[0]][0]))
        edges.extend([-1] * len(entity_change[entity[0]][0]))


        for rel in entity[2]:
            if len(rel[0]) != 0 and len(rel[1]) != 0:
                rel_label = relation_change[rel[0]]
                rel_ent_label = entity_change[rel[0]][1]
                rel_label_token = copy.deepcopy(rel[0])
                words_label = rel_ids + rel_label + tail_ids + entity_change[rel[1]][0]
                words_label_tokens = ' [relation] {} [tail] {}'.format(rel_label_token, rel[1])
                nodes.extend(
                        ([-1] * len(rel_ids)) + ([entity_change[rel[0]][1]] * len(rel_label)) + ([-1] * len(tail_ids)) + ([entity_change[rel[1]][1]] * len(
                            entity_change[rel[1]][0])))

          
                edges.extend([-1] * len(rel_ids) + [cnt_edge] * len(rel_label) + [-1] * (
                            len(tail_ids) + len(entity_change[rel[1]][0])))
                if entity_change[entity[0]][1] < len(adj_matrix) and entity_change[rel[1]][1] < len(adj_matrix):
                    if self.topology['entity-entity']:
                        adj_matrix[entity_change[entity[0]][1]][entity_change[rel[1]][1]] = 1
                        adj_matrix[entity_change[rel[1]][1]][entity_change[entity[0]][1]] = 1

                    if self.topology['entity-relation']:
                        adj_matrix[entity_change[entity[0]][1]][entity_change[rel[0]][1]] = 2
                        adj_matrix[entity_change[rel[1]][1]][entity_change[rel[0]][1]] = 2
                    
                    if self.topology['relation-entity']:
                        adj_matrix[entity_change[rel[0]][1]][entity_change[entity[0]][1]] = 3
                        adj_matrix[entity_change[rel[0]][1]][entity_change[rel[1]][1]] = 3
                        
                    if not self.topology['relation-entity'] and not self.topology['relation-relation']:
                        adj_matrix[entity_change[rel[0]][1]][entity_change[rel[0]][1]] = 10
                    
                    if not self.topology['entity-relation'] and not self.topology['entity-entity']:
                        adj_matrix[entity_change[entity[0]][1]][entity_change[entity[0]][1]] = 10
                        adj_matrix[entity_change[rel[1]][1]][entity_change[rel[1]][1]] = 10

                cnt_edge += 1
                string_label += words_label
                string_label_tokens += words_label_tokens

        assert len(string_label) == len(nodes) == len(edges)

        return string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix

    
    def relation_to_relation_fill(self, node_dict, rel_dict, adj_matrix):
        adj_matrix_temp = np.array(adj_matrix)
        rel_idx_list = []
        for rel in rel_dict.keys():
            rel_idx = node_dict[rel][1]
            rel_idx_list.append(rel_idx)
        adj_matrix_np = np.array(adj_matrix)
        adj_matrix_np_bool = (adj_matrix_np==-1)
        #reassign -1s to 0s
        adj_matrix_np[adj_matrix_np_bool] = 0
        #get squared matrix for r-r
        adj_matrix_sq = adj_matrix_np@adj_matrix_np
        
        #old adj_matrix + squared matrix only r-r
        rel_idx_list = np.array(rel_idx_list, dtype=np.intp)
        adj_matrix_temp[rel_idx_list[:,np.newaxis], rel_idx_list] = (adj_matrix_sq[rel_idx_list][:,rel_idx_list] > 0)*4
        adj_matrix_new = adj_matrix_temp.tolist()
        
        return adj_matrix_new
        
        
    def get_all_entities_per_sample(self, mark_entity_number, mark_entity, entry):
        text_entity = set()
        text_relation = set()
        for entity_id in mark_entity_number:
            entity = entry['kbs'][entity_id]
            if len(entity[0]) == 0:
                continue
            for rel in entity[2]:
                if len(rel[0]) != 0 and len(rel[1]) != 0:
                    text_relation.add(rel[0])
                    text_entity.add(rel[1])

        text_entity_list = list(text_entity)+list(text_relation)
        text_relation_list = list(text_relation)
        for entity_ele in mark_entity:
            if entity_ele in text_entity_list:
                text_entity_list.remove(entity_ele)
        
        return text_entity_list, text_relation_list

    def get_change_per_sample(self, mark_entity, text_entity, text_relation):
        # during fine-tuning, we don't mask entities or relations
        ent_change = {}
        total_entity = mark_entity + text_entity

        for ent_id in range(len(total_entity)):
            entity_toks = self.tokenizer.encode(" {}".format(total_entity[ent_id]), add_special_tokens=False)
            ent_change[total_entity[ent_id]] = [entity_toks, ent_id]
        # relation change only includes the relation tokens and ids
        rel_change = {}
        for rel_id in range(len(text_relation)):
            rel_change[text_relation[rel_id]] = self.tokenizer.encode(' {}'.format(text_relation[rel_id]),
                                                                      add_special_tokens=False)
        return ent_change, rel_change

    def truncate_pair_ar(self, a, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        # add_bos_id + graph_ids + a + text_ids + b + eos_token_id
        length_a_b = self.args.max_input_length - len(add_bos_id) - len(graph_ids) - len(text_ids) - 1
        if len(a) > length_a_b:
            a = a[:length_a_b]
            node_ids = node_ids[:length_a_b]
            edge_ids = edge_ids[:length_a_b]
        input_ids = add_bos_id + graph_ids + a + text_ids + [self.tokenizer.eos_token_id]
        input_node_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + node_ids + [-1] * (len(text_ids) + 1)
        attn_mask = [1] * len(input_ids) + [0] * (self.args.max_input_length - len(input_ids))
        input_ids += [self.tokenizer.pad_token_id] * (self.args.max_input_length - len(input_ids))
        input_node_ids += [-1] * (self.args.max_input_length - len(input_node_ids))
        assert len(input_ids) == len(attn_mask) == self.args.max_input_length == len(input_node_ids)
        return input_ids, attn_mask, input_node_ids

    def ar_prep_data(self, answers, questions, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        # add bos and eos
        decoder_label_ids = copy.deepcopy(answers)
        if len(decoder_label_ids) > self.args.max_output_length - len(add_bos_id) - 1:
            decoder_label_ids = decoder_label_ids[:(self.args.max_output_length - len(add_bos_id) - 1)]
        decoder_label_ids = add_bos_id + decoder_label_ids + [self.tokenizer.eos_token_id]
        decoder_attn_mask = [1] * len(decoder_label_ids) + [0] * (self.args.max_output_length - len(decoder_label_ids))
        decoder_label_ids += [self.tokenizer.pad_token_id] * (self.args.max_output_length - len(decoder_label_ids))
        assert len(decoder_label_ids) == self.args.max_output_length == len(decoder_attn_mask)

        input_ids, input_attn_mask, input_node_ids = self.truncate_pair_ar(questions, add_bos_id,
                                                                                           graph_ids, text_ids,
                                                                                           node_ids, edge_ids)

        return input_ids, input_attn_mask, decoder_label_ids, decoder_attn_mask, input_node_ids

    def __getitem__(self, idx):

        entry = self.data[idx]

        entities = []
        for _ in entry['kbs']:
            entities.append(_)

        strings_label = []
        node_ids = []
        edge_ids = []
        strings_label_tokens = ''

        # mark_entity: entities with KB numbers which are important for this task
        # text_entity: entities without KB numbers but only with text, which are less important
        mark_entity = [entry['kbs'][ele_entity][0] for ele_entity in entities]
        mark_entity_number = entities
        text_entity, text_relation = self.get_all_entities_per_sample(mark_entity_number, mark_entity, entry)
        entity_change, relation_change = self.get_change_per_sample(mark_entity, text_entity, text_relation)
        total_entity = mark_entity + text_entity
        adj_matrix = [[-1] * (self.args.max_node_length + 1) for _ in range(self.args.max_node_length + 1)]

        cnt_edge = 0

        if 'title' in entry:
            entity = self.knowledge[entry['title_kb_id']]
            string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix = self.linearize_v2(
                entity,
                entity_change,
                self.head_ids,
                self.rel_ids, self.tail_ids,
                relation_change, cnt_edge, adj_matrix)

            strings_label += string_label
            strings_label_tokens += string_label_tokens

        for i, entity_id in enumerate(entities):
            entity = entry['kbs'][entity_id]
            string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix = self.linearize_v2(
                entity,
                entity_change,
                self.head_ids,
                self.rel_ids, self.tail_ids,
                relation_change, cnt_edge, adj_matrix)
            
            strings_label += string_label
            strings_label_tokens += string_label_tokens
            node_ids += nodes
            edge_ids += edges
            
        if self.topology['relation-relation']:
            adj_matrix = self.relation_to_relation_fill(entity_change, relation_change, adj_matrix)
        

        words_label_ids, words_label_tokens, words_input_ids, words_input_tokens = [], '', [], ''
        current_text = random.choice(entry['text'])
       
        for word in current_text.split():
            word_label_ids = self.tokenizer.encode(" {}".format(word), add_special_tokens=False)
            word_label_tokens = copy.deepcopy(word)

            words_label_ids += word_label_ids
            words_label_tokens += ' ' + word_label_tokens

        input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask, input_node_ids_ar = \
            self.ar_prep_data(words_label_ids, strings_label, self.add_bos_id, self.graph_ids,
                              self.text_ids, node_ids, edge_ids)

        node_length_ar = max(input_node_ids_ar) + 1        

        def masked_fill(src, masked_value, fill_value):
            return [src[src_id] if src[src_id] != masked_value and src[src_id] < fill_value else fill_value for src_id
                    in range(len(src))]

        input_node_ids_ar = masked_fill(input_node_ids_ar, -1, self.args.max_node_length)

        def masked_fill_matrix(adj_matrix_input, masked_value, fill_value):
            adj_matrix_tmp = copy.deepcopy(adj_matrix_input)
            for a_id in range(len(adj_matrix_tmp)):
                for b_id in range(len(adj_matrix_tmp)):
                    if adj_matrix_tmp[a_id][b_id] == masked_value or adj_matrix_tmp[a_id][b_id] > fill_value:
                        adj_matrix_tmp[a_id][b_id] = fill_value
            return adj_matrix_tmp

        adj_matrix_ar = masked_fill_matrix(adj_matrix, -1, self.args.max_edge_length)

        assert len(input_ids_ar) == len(attn_mask_ar) == self.args.max_input_length == len(input_node_ids_ar)
        assert len(decoder_label_ids) == len(decoder_attn_mask) == self.args.max_output_length

        input_ids_ar = torch.LongTensor(input_ids_ar)
        attn_mask_ar = torch.LongTensor(attn_mask_ar)
        decoder_label_ids = torch.LongTensor(decoder_label_ids)
        decoder_attn_mask = torch.LongTensor(decoder_attn_mask)
        input_node_ids_ar = torch.LongTensor(input_node_ids_ar)
        node_length_ar = torch.LongTensor([node_length_ar])
        adj_matrix_ar = torch.LongTensor(adj_matrix_ar)
       
        num_triples = strings_label_tokens.count("[relation]")
        return input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask, \
               input_node_ids_ar, node_length_ar, adj_matrix_ar, num_triples        

        
        
        
        
class EventDataset(Dataset):
    def __init__(self, logger, args, data_path, tokenizer, mode):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.topology = {"entity-entity": args.entity_entity, 
                           "entity-relation": args.entity_relation,
                           "relation-entity": args.relation_entity,
                           "relation-relation": args.relation_relation
                          }                        
                        
                        
        with open(self.data_path+ '.source') as f:
            source_kgs = [sample.rstrip('\n') for sample in f]
       
        with open(self.data_path+ '.target.tok') as f:
            target_texts = [sample.rstrip('\n') for sample in f]
        
        self.data = list(zip(source_kgs, target_texts))
        print("Total samples = {}".format(len(self.data)))

        if args.debug:
            self.data = self.data[:1000]
        assert type(self.data) == list
        self.args = args
        self.data_type = mode
        self.metric = "BLEU"
        self.head_ids, self.rel_ids, self.tail_ids = self.tokenizer.encode(' <S>', add_special_tokens=False), \
                                                     self.tokenizer.encode(' <P>', add_special_tokens=False), \
                                                     self.tokenizer.encode(' <O>', add_special_tokens=False)
        self.graph_ids, self.text_ids = self.tokenizer.encode(' [graph]', add_special_tokens=False), \
                                        self.tokenizer.encode(' [text]', add_special_tokens=False)

        if self.args.model_name == "bart":
            self.mask_token = self.tokenizer.mask_token
            self.mask_token_id = self.tokenizer.mask_token_id
        else:
            self.mask_token = self.tokenizer.additional_special_tokens[0]
            self.mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.additional_special_tokens[0])

        if self.args.model_name == "bart":
            if self.args.append_another_bos:
                self.add_bos_id = [self.tokenizer.bos_token_id] * 2
            else:
                self.add_bos_id = [self.tokenizer.bos_token_id]
        else:
            self.add_bos_id = []

    def __len__(self):
        return len(self.data)
    
    def graph_size(self,idx):
        entry = self.data[idx]
        kg = entry[0]
        
        kg_list = []
        triple_list = kg.split('<S>')
        triple_list = [triple_list[0]] + ['<S>'+triple for triple in triple_list[1:]]
        triple_list = list(filter(None,triple_list))
        for triple in triple_list:
            head = re.search('<S>(.*)<P>', triple).group(1).strip()
            rel = re.search('<P>(.*)<O>', triple).group(1).strip()
            tail = re.search('<O>(.*)', triple).group(1).strip()
            kg_list.append([head,rel,tail])
        
        

        strings_label = []
        node_ids = []
        edge_ids = []
        strings_label_tokens = ''

        
        text_entity, text_relation = self.get_all_entities_per_sample(kg_list)
        entity_change, relation_change = self.get_change_per_sample(text_entity, text_relation)
        return len(entity_change)

    def graph_linearize(self, triple, entity_change, head_ids, rel_ids, tail_ids,
                        relation_change, cnt_edge, adj_matrix):
        # string_label: encoder ids
        # string_label_tokens: encoder tokens
        if len(triple[0]) == 0:
            return [], '', [], [], cnt_edge, adj_matrix
        nodes, edges = [], []
        string_label = copy.deepcopy(head_ids)
        string_label_tokens = ' <S>'
        nodes.extend([-1] * len(string_label))
        edges.extend([-1] * len(string_label))


        string_label += entity_change[triple[0]][0]
        string_label_tokens += ' {}'.format(triple[0])
        nodes.extend([entity_change[triple[0]][1]] * len(entity_change[triple[0]][0]))
        edges.extend([-1] * len(entity_change[triple[0]][0]))


        if len(triple[1]) != 0 and len(triple[2]) != 0:
            rel_label = relation_change[triple[1]]
            rel_ent_label = entity_change[triple[1]][1]
            rel_label_token = copy.deepcopy(triple[1])
            words_label = rel_ids + rel_label + tail_ids + entity_change[triple[2]][0]
            words_label_tokens = ' <P> {} <O> {}'.format(rel_label_token, triple[2])
            nodes.extend(
                    ([-1] * len(rel_ids)) + ([entity_change[triple[1]][1]] * len(rel_label)) + ([-1] * len(tail_ids)) + ([entity_change[triple[2]][1]] * len(
                        entity_change[triple[2]][0])))
            edges.extend([-1] * len(rel_ids) + [cnt_edge] * len(rel_label) + [-1] * (
                        len(tail_ids) + len(entity_change[triple[2]][0])))
            if entity_change[triple[0]][1] < len(adj_matrix) and entity_change[triple[2]][1] < len(adj_matrix):


                if self.topology['entity-entity']:
                    adj_matrix[entity_change[triple[0]][1]][entity_change[triple[2]][1]] = 1
                    adj_matrix[entity_change[triple[2]][1]][entity_change[triple[0]][1]] = 1

                if self.topology['entity-relation']:
                    adj_matrix[entity_change[triple[0]][1]][entity_change[triple[1]][1]] = 2
                    adj_matrix[entity_change[triple[2]][1]][entity_change[triple[1]][1]] = 2

                if self.topology['relation-entity']:
                    adj_matrix[entity_change[triple[1]][1]][entity_change[triple[0]][1]] = 3
                    adj_matrix[entity_change[triple[2]][1]][entity_change[triple[1]][1]] = 3
                    
                if not self.topology['relation-entity'] and not self.topology['relation-relation']:
                    adj_matrix[entity_change[triple[1]][1]][entity_change[triple[1]][1]] = 10

                if not self.topology['entity-relation'] and not self.topology['entity-entity']:
                    adj_matrix[entity_change[triple[0]][1]][entity_change[triple[0]][1]] = 10
                    adj_matrix[entity_change[triple[2]][1]][entity_change[triple[2]][1]] = 10

            cnt_edge += 1
            string_label += words_label
            string_label_tokens += words_label_tokens

        assert len(string_label) == len(nodes) == len(edges)

        return string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix

    def relation_to_relation_fill(self, node_dict, rel_dict, adj_matrix):
        adj_matrix_temp = np.array(adj_matrix)
        rel_idx_list = []
        for rel in rel_dict.keys():
            rel_idx = node_dict[rel][1]
            rel_idx_list.append(rel_idx)
        adj_matrix_np = np.array(adj_matrix)
        adj_matrix_np_bool = (adj_matrix_np==-1)
        #reassign -1s to 0s
        adj_matrix_np[adj_matrix_np_bool] = 0
        #get squared matrix for r-r
        adj_matrix_sq = adj_matrix_np@adj_matrix_np
        
        #old adj_matrix + squared matrix only r-r
        rel_idx_list = np.array(rel_idx_list, dtype=np.intp)
        adj_matrix_temp[rel_idx_list[:,np.newaxis], rel_idx_list] = (adj_matrix_sq[rel_idx_list][:,rel_idx_list] > 0)*4
        adj_matrix_new = adj_matrix_temp.tolist()
        
        return adj_matrix_new
    
    def get_all_entities_per_sample(self, triple_list):
        text_entity = set()
        text_relation = set()
        for triple in triple_list:
            if len(triple[0]) == 0:
                continue
            if len(triple[1]) != 0 and len(triple[2]) != 0:
                text_relation.add(triple[1])
                text_entity.add(triple[0])
                text_entity.add(triple[2])
                
        text_entity_list = list(text_entity)+list(text_relation)
        text_relation_list = list(text_relation)
        
        return text_entity_list, text_relation_list

    def get_change_per_sample(self, text_entity, text_relation):
        # during fine-tuning, we don't mask entities or relations
        ent_change = {}
        total_entity = text_entity

        for ent_id in range(len(total_entity)):
            entity_toks = self.tokenizer.encode(" {}".format(total_entity[ent_id]), add_special_tokens=False)
            ent_change[total_entity[ent_id]] = [entity_toks, ent_id]
            
        # relation change only includes the relation tokens and ids
        rel_change = {}
        for rel_id in range(len(text_relation)):
            rel_change[text_relation[rel_id]] = self.tokenizer.encode(' {}'.format(text_relation[rel_id]),
                                                                      add_special_tokens=False)

        return ent_change, rel_change

    def truncate_pair_ar(self, a, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        # add_bos_id + graph_ids + a + text_ids + b + eos_token_id
        length_a_b = self.args.max_input_length - len(add_bos_id) - len(graph_ids) - len(text_ids) - 1
        if len(a) > length_a_b:
            a = a[:length_a_b]
            node_ids = node_ids[:length_a_b]
            edge_ids = edge_ids[:length_a_b]
        input_ids = add_bos_id + graph_ids + a + text_ids + [self.tokenizer.eos_token_id]
        input_node_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + node_ids + [-1] * (len(text_ids) + 1)
        attn_mask = [1] * len(input_ids) + [0] * (self.args.max_input_length - len(input_ids))
        input_ids += [self.tokenizer.pad_token_id] * (self.args.max_input_length - len(input_ids))
        input_node_ids += [-1] * (self.args.max_input_length - len(input_node_ids))
        assert len(input_ids) == len(attn_mask) == self.args.max_input_length == len(input_node_ids)
        return input_ids, attn_mask, input_node_ids

    def ar_prep_data(self, answers, questions, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        # add bos and eos
        decoder_label_ids = copy.deepcopy(answers)
        if len(decoder_label_ids) > self.args.max_output_length - len(add_bos_id) - 1:
            decoder_label_ids = decoder_label_ids[:(self.args.max_output_length - len(add_bos_id) - 1)]
        decoder_label_ids = add_bos_id + decoder_label_ids + [self.tokenizer.eos_token_id]
        decoder_attn_mask = [1] * len(decoder_label_ids) + [0] * (self.args.max_output_length - len(decoder_label_ids))
        decoder_label_ids += [self.tokenizer.pad_token_id] * (self.args.max_output_length - len(decoder_label_ids))
        assert len(decoder_label_ids) == self.args.max_output_length == len(decoder_attn_mask)

        input_ids, input_attn_mask, input_node_ids = self.truncate_pair_ar(questions, add_bos_id,
                                                                                           graph_ids, text_ids,
                                                                                           node_ids, edge_ids)

        return input_ids, input_attn_mask, decoder_label_ids, decoder_attn_mask, input_node_ids

    def __getitem__(self, idx):
        entry = self.data[idx]
        kg = entry[0]

        kg_list = []
        triple_list = kg.split('<S>')
        triple_list = [triple_list[0]] + ['<S>'+triple for triple in triple_list[1:]]
        triple_list = list(filter(None,triple_list))
        for triple in triple_list:
            head = re.search('<S>(.*)<P>', triple).group(1).strip()
            rel = re.search('<P>(.*)<O>', triple).group(1).strip()
            tail = re.search('<O>(.*)', triple).group(1).strip()
            kg_list.append([head,rel,tail])
        
        strings_label = []
        node_ids = []
        edge_ids = []
        strings_label_tokens = ''

        
        text_entity, text_relation = self.get_all_entities_per_sample(kg_list)
        entity_change, relation_change = self.get_change_per_sample(text_entity, text_relation)
        adj_matrix = [[-1] * (self.args.max_node_length + 1) for _ in range(self.args.max_node_length + 1)]

        cnt_edge = 0

        for i, triple in enumerate(kg_list):
            string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix = self.graph_linearize(
                triple,
                entity_change,
                self.head_ids,
                self.rel_ids, self.tail_ids,
                relation_change, cnt_edge, adj_matrix)
            
            strings_label += string_label
            strings_label_tokens += string_label_tokens
            node_ids += nodes
            edge_ids += edges
        
        if self.topology['relation-relation']:
            adj_matrix = self.relation_to_relation_fill(entity_change, relation_change, adj_matrix)
        
        words_label_ids, words_label_tokens, words_input_ids, words_input_tokens = [], '', [], ''
        current_text = entry[1]
       
        for word in current_text.split():
            word_label_ids = self.tokenizer.encode(" {}".format(word), add_special_tokens=False)
            word_label_tokens = copy.deepcopy(word)

            words_label_ids += word_label_ids
            words_label_tokens += ' ' + word_label_tokens

        input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask, input_node_ids_ar = \
            self.ar_prep_data(words_label_ids, strings_label, self.add_bos_id, self.graph_ids,
                              self.text_ids, node_ids, edge_ids)

        node_length_ar = max(input_node_ids_ar) + 1        

        def masked_fill(src, masked_value, fill_value):
            return [src[src_id] if src[src_id] != masked_value and src[src_id] < fill_value else fill_value for src_id
                    in range(len(src))]

        input_node_ids_ar = masked_fill(input_node_ids_ar, -1, self.args.max_node_length)

        def masked_fill_matrix(adj_matrix_input, masked_value, fill_value):
            adj_matrix_tmp = copy.deepcopy(adj_matrix_input)
            for a_id in range(len(adj_matrix_tmp)):
                for b_id in range(len(adj_matrix_tmp)):
                    if adj_matrix_tmp[a_id][b_id] == masked_value or adj_matrix_tmp[a_id][b_id] > fill_value:
                        adj_matrix_tmp[a_id][b_id] = fill_value
            return adj_matrix_tmp

        adj_matrix_ar = masked_fill_matrix(adj_matrix, -1, self.args.max_edge_length)

        assert len(input_ids_ar) == len(attn_mask_ar) == self.args.max_input_length == len(input_node_ids_ar)
        assert len(decoder_label_ids) == len(decoder_attn_mask) == self.args.max_output_length

        input_ids_ar = torch.LongTensor(input_ids_ar)
        attn_mask_ar = torch.LongTensor(attn_mask_ar)
        decoder_label_ids = torch.LongTensor(decoder_label_ids)
        decoder_attn_mask = torch.LongTensor(decoder_attn_mask)
        input_node_ids_ar = torch.LongTensor(input_node_ids_ar)
        node_length_ar = torch.LongTensor([node_length_ar])
        adj_matrix_ar = torch.LongTensor(adj_matrix_ar)
       
        return input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask, \
               input_node_ids_ar, node_length_ar, adj_matrix_ar


def evaluate_bleu(data_ref, data_sys):
    coco_eval = run_coco_eval(data_ref, data_sys)
    scores = {metric: score for metric, score in list(coco_eval.eval.items())}
    return scores["Bleu_4"]
