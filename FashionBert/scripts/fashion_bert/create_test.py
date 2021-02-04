import random
import numpy as np
import csv
import datetime
import argparse

# 0: feature
# 1: image_mask SAME
# 2: input_ids (text_ids)
# 3: input_masks SAME
# 4: segment_ids SAME
# 5: nx_sent label
# 6: prod_desc
# 7: text_prod_id
# 8: image_prod_id
# 9: prod_img_id


def main(args):
    folder = "eval_img2txt" if args.type == "i2t" else "eval_txt2img"
    file = "{}/eval_img2txt/{}".format(args.bert_path, args.name_data)
    outfile = "{}/{}/{}".format(args.bert_path, folder, args.name_out)

    data_im, data_text, ids = get_data(file)

    if args.split == "test":
        test(args, data_im, data_text, ids, outfile)
    elif args.split == "train":
        train(args, data_im, data_text, ids, outfile)

def train(args, data_im, data_text, ids, outfile):
    with open(outfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')

        for i in range(len(ids)):

            print("{}: [{}/{}]".format(datetime.datetime.now().time(), i, len(ids)))

            query_id = ids[i]

            filtered_ids = list_remove(ids, i)
            random_ids = random.sample(filtered_ids, 1)

            length = length_cap(data_text[query_id][0])

            # create all the data needed
            image_mask = ','.join(map(str, np.ones(64, dtype=int)))
            masked_patch_pos = mask_pp()
            segment_ids = ','.join(map(str, np.zeros(64, dtype=int)))
            masked_lm_positions, masked_lm_ids, masked_lm_weights = mask_lm(length, data_text[query_id][0])
            input_mask = create_mask(length)

            # write true pair
            writer.writerow((data_im[query_id], image_mask, masked_patch_pos,
                            data_text[query_id][0], input_mask, segment_ids,
                            masked_lm_positions, masked_lm_ids, masked_lm_weights,
                            int(1)))

            # write 1 false pairs
            for j in range(len(random_ids)):
                target_id = random_ids[j]
                length = length_cap(data_text[target_id][0])

                masked_patch_pos = mask_pp()
                masked_lm_positions, masked_lm_ids, masked_lm_weights = mask_lm(length, data_text[target_id][0])
                input_mask = create_mask(length)

                # write true pair
                writer.writerow((data_im[query_id], image_mask, masked_patch_pos,
                                data_text[target_id][0], input_mask, segment_ids,
                                masked_lm_positions, masked_lm_ids, masked_lm_weights,
                                int(0)))



def mask_lm(length, caption_ids):
    caption_ids = caption_ids.split(",")
    mask_pos = np.zeros(10, dtype=int)
    mask_ids = np.zeros(10, dtype=int)
    mask_weights = np.zeros(10)

    count = 0
    for i in range(1, length-1):
        if count > 9:
            break

        if random.random() < 0.15:
            mask_pos[count] = i
            mask_ids[count] = int(caption_ids[i])
            mask_weights[count] = 1.0
            count += 1

    # when zero masking, mask at least one
    if count == 0:
        n = random.randint(1,length-1)
        mask_pos[0] = n
        mask_ids[0] = int(caption_ids[n])
        mask_weights[0] = 1.0

    str_pos = ','.join(map(str, mask_pos))
    str_ids = ','.join(map(str, mask_ids))
    str_weights = ','.join(map(str, mask_weights))

    return str_pos, str_ids, str_weights

def mask_pp():
    data = [i for i in range(64)]
    random.shuffle(data)
    sample = random.sample(data, 5)
    return ','.join(map(str, sample))

def test(args, data_im, data_text, ids, outfile):
    if args.half == "first":
        start = 0
        end = 500
    else:
        start = 500
        end = 1000

    with open(outfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')

        for i in range(start, end):

            print("{}: [{}/{}]".format(datetime.datetime.now().time(), i, len(data_im)))

            query_id = ids[i]

            filtered_ids = list_remove(ids, i)
            random_ids = random.sample(filtered_ids, 100)

            image_mask = ','.join(map(str, np.ones(64, dtype=int)))
            segment_ids = ','.join(map(str, np.zeros(64, dtype=int)))
            length = length_cap(data_text[query_id][0])
            input_mask = create_mask(length)

            # write true pair
            writer.writerow((data_im[query_id], image_mask, data_text[query_id][0], input_mask, segment_ids, int(1), data_text[query_id][1], query_id, query_id, str(query_id)+"_0"))


            # write 100 false pairs
            for j in range(len(random_ids)):
                target_id = random_ids[j]
                if args.type == "i2t":
                    length = length_cap(data_text[target_id][0])
                    input_mask = create_mask(length)
                    writer.writerow((data_im[query_id], image_mask, data_text[target_id][0], input_mask, segment_ids, int(0), data_text[target_id][1], target_id, query_id, str(query_id)+"_0"))
                else:
                    writer.writerow((data_im[target_id], image_mask, data_text[query_id][0], input_mask, segment_ids, int(0), data_text[query_id][1], query_id, target_id, str(target_id)+"_0"))

def create_mask(length):
    input_mask = np.concatenate([np.ones(length, dtype=int), np.zeros(64-length, dtype=int)])
    str_input_mask = ','.join(map(str, input_mask))
    return str_input_mask

def length_cap(text):
    temp = text.split(",")
    zeros = 0
    non_zeros = 0
    for i in range(len(temp)):
        if int(temp[i].strip()) == 0:
            zeros += 1
        else:
            non_zeros += 1
    return non_zeros

def get_data(file):
    data_im = {}
    data_text = {}
    ids = []
    count = 0
    with open(file) as infile:
        for line in infile:
            temp = line.split("\t")
            print(temp[5])
            print(temp[6])
            print(temp[7])
            id = int(temp[7])
            # id = count
            # count += 1
            ids.append(id)
            data_im[id] = temp[0]
            data_text[id] = (temp[2], temp[6])
    return data_im, data_text, ids

def list_remove(list, index):
    return list[:index] + list[index+1 :]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate features from image')
    parser.add_argument('--name_data',help='ldocation data file', default="data_caption_train.txt", type=str)
    parser.add_argument('--name_out',help='location data out-file', default="data_caption_TEST_1.txt", type=str)
    parser.add_argument('--bert_path',help='path to fashionbert', default=".", type=str)
    parser.add_argument('--type',help='t2i or i2t', default="i2t", type=str)
    parser.add_argument('--half',help='first or second half', default="first", type=str)
    parser.add_argument('--split',help='train, val, test', default="test", type=str)

    args = parser.parse_args()

    main(args)
