import tensorflow as tf

# file = ["eval_img2txt/fashionbert-fashiongen-patch-eval-img2txt__097eaabf2d1e4464b88453bc7dfc8878","eval_img2txt/fashionbert-fashiongen-patch-eval-img2txt__097eaabf2d1e4464b88453bc7dfc8878"]
#
# a = tf.constant(file)
# print(a)
# d = tf.data.Dataset.from_tensor_slices(a)
# print(d)
# d = d.shard(len(self.worker_hosts.split(',')), self.task_index)
# d = d.repeat(1)
# cycle_length = min(4, len(self.input_fps))



# file = "eval_img2txt/fashionbert-fashiongen-patch-eval-img2txt__097eaabf2d1e4464b88453bc7dfc8878"
file = "eval_img2txt/data_captions_None.txt"
count = 0
with open(file) as infile:
    for line in infile:
        k = line.split("\t")
        for i in range(len(k)):
            print(k[i])
        print("length", len(k[3].split(",")))
        count += 1
        if count == 3:
            exit()
