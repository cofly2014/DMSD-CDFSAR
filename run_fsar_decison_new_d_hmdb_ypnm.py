import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import numpy as np
import argparse

import pickle
from utils import print_and_log, get_log_files, TestAccuracies, loss, aggregate_accuracy#, send_email

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from torch.optim.lr_scheduler import MultiStepLR
import video_reader_cross_res as  video_reader
import random
import logging

from models.cross_domain_fsar_resnet_new_d import CROSS_DOMAIN_FSAR


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


# logger for training accuracies
train_logger = setup_logger('Training_accuracy', 'runs_trms/train_output.log')
# logger for evaluation accuracies
eval_logger = setup_logger('Evaluation_accuracy', 'runs_trms/eval_output.log')

###############################################3#
# 增加随机数种子
# setting up seeds
manualSeed = random.randint(1, 10000)
#manualSeed = 1888
#manualSeed = 1896
manualSeed = 2024
print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
################################################3
"""
Command line parser
"""


def parse_command_line():
    parser = argparse.ArgumentParser()
    # 经常需要修改
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--tasks_per_batch", type=int, default=16, help="Number of tasks between parameter optimizations.")
    # 经常需要修改
    parser.add_argument("--training_iterations", "-i", type=int, default=23020, help="Number of meta-training iterations.")
    # 经常需要修改
    parser.add_argument("--num_test_tasks", type=int, default=1000, help="number of random tasks to test on.")
    parser.add_argument('--test_iters', nargs='+', type=int, default=[20000], help='iterations to test at. Default is for ssv2 otam split.')

    #元训练阶段一个episode中抽样多少个未标记的target doamin的样本
    parser.add_argument("--target_unlabeled_num", type=int, default=25, help="Way of each task.")

    parser.add_argument("--way", type=int, default=5, help="Way of each task.")
    parser.add_argument("--shot", type=int, default=5, help="Shots per class.")
    parser.add_argument("--query_per_class", type=int, default=5, help="Target samples (i.e. queries) per class used for training.")
    parser.add_argument("--query_per_class_test", type=int, default=1,  help="Target samples (i.e. queries) per class used for testing.")
    parser.add_argument("--print_freq", type=int, default=100, help="print and log every n iterations.")  # 1000

    parser.add_argument("--num_workers", type=int, default=4, help="Num dataloader workers.")
    parser.add_argument("--method", choices=["resnet18", "resnet34", "resnet50"], default="resnet18", help="method")
    parser.add_argument("--opt", choices=["adam", "sgd"], default="adam", help="Optimizer")
    parser.add_argument("--trans_dropout", type=int, default=0.1, help="Transformer dropout")
    # 可能需要修改
    parser.add_argument("--save_freq", type=int, default=200, help="Number of iterations between checkpoint saves.")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size to the CNN after cropping.")

    parser.add_argument("--scratch", choices=["bc", "bp", "new"], default="bp", help="directory containing dataset, splits, and checkpoint saves.")
    # 经常需要修改
    parser.add_argument("--gpus_use", default=[0, 1, 2, 3], help="GPUs No. to split the ResNet over")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to split the ResNet over")

    parser.add_argument("--debug_loader", default=False, action="store_true", help="Load 1 vid per class for debugging")
    # 可能需要修改 不同的split这里的编号不一样
    parser.add_argument("--split", type=int, default=3, help="Dataset split.")
    parser.add_argument('--sch', nargs='+', type=int, help='iters to drop learning rate', default=[1600000])
    #####################数据集和pt保存路径#############
    #parser.add_argument("--dataset", choices=["ssv2", "ssv2_cmn", "kinetics", "hmdb", "ucf"], default="hmdb", help="Dataset to use.")
    parser.add_argument("--source_dataset", choices=["ssv2", "ssv2_cmn", "kinetics", "hmdb", "ucf", "diving48", "rareAct"], default="kinetics", help="Dataset to use.")
    parser.add_argument("--target_dataset", choices=["ssv2", "ssv2_cmn", "kinetics", "hmdb", "ucf", "diving48", "rareAct"], default="hmdb",       help="Dataset to use.")

    parser.add_argument("--checkpoint_dir", "-c", default="./checkpoint_hmdb_ypnm/", help="Directory to save checkpoint to.")
    parser.add_argument("--test_model_path", "-m", default="./checkpoint_hmdb_ypnm/", help="Path to model to load and test.")

    parser.add_argument("--resume_checkpoint_iter", type=int, default=10000, help="Path to model to resume.")
    parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False, action="store_true", help="Restart from latest checkpoint.")

    parser.add_argument("--test_checkpoint_iter", type=int, default=20000, help="Path to model to load and test.")
    parser.add_argument("--test_model_only", type=bool, default=False, help="Only testing the model from the given checkpoint")

    #####################下面的是消融参数##############
    parser.add_argument("--lambdas", type=int, default=[1, 0.5, 0, 0, 0, 0], help="temporal_set")  #1是监督分类loss 2是蒸馏loss

    parser.add_argument("--TRANSFORMER_DEPTH", type=int, default=1, help="transformer depth")  # 1是监督分类loss 2是蒸馏loss
    parser.add_argument("--SINGLE_DIRECT", type=int, default=False, help="otam direction")  # 1是监督分类loss 2是蒸馏loss

    parser.add_argument("--seq_len", type=int, default=8, help="Frames per video.")
    parser.add_argument('--sub_seq_len', type=int, default=8, help='the length of sub sequence')
    parser.add_argument('--sub_seq_num', type=int, default=1, help='the length of sub sequence')
    parser.add_argument('--start_cross', type=int, default= 10000, help='the time to start meta-learning step2')

    parser.add_argument('--class_num', type=int, default=64, help='the number of class catetories in the training stage')

    args = parser.parse_args()

    # 数据集存储在不同的地方 移动硬盘，服务器，本地PC
    if args.scratch == "bc":
        args.num_gpus = 2
        args.scratch = "./"
        args.scratch_data = "/mnt/mydata/video-mini-frames/"
    elif args.scratch == "bp":
        args.num_gpus = 3
        # this is low becuase of RAM constraints for the data loader
        args.num_workers = 3
        args.scratch = "./"
        args.scratch_data = "/home/guofei/mydata/video-mini-frames/"
        #args.scratch_data = "/local/wangq/data/video-mini-frames/"

        args.unlabel_scratch = "./"
        #args.unlabel_scratch_data = "/home/guofei/mydata/video-mini-frames/"
        args.unlabel_scratch_data = "/data1/wangq/data/video-mini-frames/"
    elif args.scratch == "new":
        args.num_workers = 2
        args.num_gpus = 1
        args.scratch = "./"
        args.scratch_data = "E:/mydata/video-mini-frames/"

    if args.checkpoint_dir == None:
        print("need to specify a checkpoint dir")
        exit(1)

    if (args.method == "resnet50") or (args.method == "resnet34"):
        args.img_size = 224

    if args.method == "resnet50":
        args.trans_linear_in_dim = 2048
    else:
        args.trans_linear_in_dim = 512
    #############################################################################################
    #源域
    if args.source_dataset == "kinetics":
        args.trainlist = os.path.join(args.scratch, "splits/kinetics_CMN")
        args.scratch_data = "/home/guofei/mydata/video-mini-frames/"
        args.scratch_data = "/data1/wangq/data/video-mini-frames/"
        args.path = os.path.join(args.scratch_data, "kinetics-mini-frames/")
    elif args.source_dataset == "kinetics_100":
        args.trainlist = os.path.join(args.scratch, "splits/kinetics_100")
        args.scratch_data = "/home/guofei/mydata/video-origin-frames/"
        args.path = os.path.join(args.scratch_data, "kinetics-frames/train_400/")
    elif args.source_dataset == "kinetics_400":
        args.trainlist = os.path.join(args.scratch, "splits/kinetics_400")
        args.scratch_data = "/home/guofei/mydata/video-origin-frames/"
        args.path = os.path.join(args.scratch_data, "kinetics-frames/train_400/")
    #############################################################################################
    #############################################################################################
    #目标域
    if args.target_dataset == "ssv2":
        args.scratch_data = "/mnt/mydata/video-mini-frames/"
        args.scratch_data = "/data1/wangq/data/video-mini-frames/"
        args.target_traintestlist = os.path.join(args.scratch, "splits/ssv2_OTAM")
        args.target_path = os.path.join(args.scratch_data, "mini-ssv2-frames-number/")
    elif args.target_dataset == "kinetics":
        args.target_traintestlist = os.path.join(args.scratch, "splits/kinetics_CMN")
        args.target_path = os.path.join(args.scratch_data, "kinetics-mini-frames/")
    elif args.target_dataset == "ucf":
        args.scratch_data = "/mnt/mydata/video-mini-frames/"
        args.target_traintestlist = os.path.join(args.scratch, "splits/ucf_ARN")
        args.target_path = os.path.join(args.scratch_data, "UCF101_frames/")
    elif args.target_dataset == "hmdb":
        args.scratch_data = "/mnt/mydata/video-mini-frames/"
        args.target_traintestlist = os.path.join(args.scratch, "splits/hmdb_ARN")
        args.target_path = os.path.join(args.scratch_data, "HMDB_51/")
    elif args.target_dataset == "ssv2_cmn":
        args.scratch_data = "/mnt/mydata/video-mini-frames/"
        args.target_traintestlist = os.path.join(args.scratch, "splits/ssv2_CMN")
        args.target_path = os.path.join(args.scratch_data, "mini-ssv2-frames/")
    elif args.target_dataset == "diving48":
        args.scratch_data = "/mnt/mydata/video-origin-frames/"
        args.target_traintestlist = os.path.join(args.scratch, "splits/diving48")
        args.target_path = os.path.join(args.scratch_data, "diving48-frames-v1/")
    if args.target_dataset == "ssv2_mimi":
        args.target_traintestlist = os.path.join(args.scratch, "splits/ssv2_mini")
        args.target_path = os.path.join(args.scratch_data, "mini-ssv2-frames-number/")

    if args.target_dataset == "rareAct":
        args.scratch_data = "/home/guofei/dataset_download/"
        args.target_traintestlist = os.path.join(args.scratch, "splits/rareAction")
        args.target_path = os.path.join(args.scratch_data, "RareAct_cut_frames/")


    #############################################################################################
    with open("args.pkl", "wb") as f:
        pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)
    return args
##################################################

def main():
    learner = Learner()
    learner.run()


class Learner:

    def __init__(self):
        self.args = parse_command_line()
        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final = get_log_files(self.args.checkpoint_dir, self.args.resume_from_checkpoint, False)
        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)
        ##############################做试验，抢资源时候需要修改的的地方###############################
        # 初始化主模型
        self.model = self.init_model()
        ################################################################################
        self.train_set, self.validation_set, self.test_set = self.init_data()
        ################################################################################
        self.vd = video_reader.VideoDataset(self.args)
        self.video_loader = torch.utils.data.DataLoader(self.vd, batch_size=1, num_workers=self.args.num_workers)
        # 进行loss function 以及accurary_fn的定义，这两个方法定义在untils.py中
        self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        # 根据参数定义优化器
        if self.args.opt == "adam":
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            # frozen
            '''
            for p in self.model.parameters():
                p.requires_grad = True
            for p in self.model.backbone.parameters():
                p.requires_grad = True

            feature_params = filter(lambda p: p.requires_grad, self.model.parameters())
            '''
            #self.optimizer = torch.optim.Adam(feature_params, lr=self.args.learning_rate)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.opt == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        #
        self.test_accuracies = TestAccuracies(self.test_set)
        # 其作用在于对optimizer中的学习率进行更新、调整，更新的方法是scheduler.step()。
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.sch, gamma=1)

        self.start_iteration = 0
        # 根据参数，如何不是测试模式且,需要加载模型，则加载模型
        if self.args.resume_from_checkpoint and not self.args.test_model_only:
            self.load_checkpoint(self.args.resume_checkpoint_iter)
        self.optimizer.step()
        # 用0梯度，初始化优化器
        self.optimizer.zero_grad()

    def init_model(self):
        model = CROSS_DOMAIN_FSAR(self.args)
        model = model.cuda()
        if self.args.num_gpus > 1:
            model.distribute_model()
        return model

    def init_data(self):
        train_set = [self.args.source_dataset]
        validation_set = [self.args.target_dataset]
        test_set = [self.args.target_dataset]
        return train_set, validation_set, test_set

    def run(self):

        ###########################################
        if self.args.test_model_only:
            print("Model being tested at path: " + self.args.test_model_path + "  " + str(self.args.test_checkpoint_iter))
            self.load_checkpoint(self.args.test_checkpoint_iter)
            accuracy_dict = self.test(1)
            print(accuracy_dict)
            return
        ##########################################

        ##########################################
        total_iterations = self.args.training_iterations
        iteration = self.start_iteration
        self.train(iteration, total_iterations)
        ##########################################

        ##########################################
        # save the final model
        torch.save(self.model.state_dict(), self.checkpoint_path_final)
        self.logfile.close()
        ##########################################

    def train(self, iteration, total_iterations):
        # 训练精确度,损失
        train_accuracies = []
        losses = []
        # 总的迭代次数
        # 训练每次循环就是一个episode
        for task_dict in self.video_loader:
            if iteration >= total_iterations:
                break
            iteration += 1
            torch.set_grad_enabled(True)
            print("start to train iteration: " + str(iteration))
            task_loss, task_accuracy = self.train_task(task_dict, iteration)
            print("finish to train iteration: " + str(iteration))
            train_accuracies.append(task_accuracy)
            losses.append(task_loss)
            #optimize  每tasks_per_batch次iteration进行一次网络参数更新, 模型开始新一轮的迭代
            if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
                self.optimizer.step()
                self.optimizer.zero_grad() #每tasks_per_batch 次 梯度值归零
            self.scheduler.step()
            # 每print_freq个iteration 会打印一次日志, 计算一次平均loss和平均的training accuracy
            if (iteration + 1) % self.args.print_freq == 0:
                # console log  log file
                print_and_log(self.logfile, 'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'.format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(), torch.Tensor(train_accuracies).mean().item()))
                train_logger.info("For Task: {0}, the training loss is {1} and Training Accuracy is {2}".format(iteration + 1, torch.Tensor(losses).mean().item(), torch.Tensor(train_accuracies).mean().item()))

                avg_train_acc = torch.Tensor(train_accuracies).mean().item()
                avg_train_loss = torch.Tensor(losses).mean().item()

                train_accuracies = []
                losses = []
            # 每save_freq个episode保存一次模型，最有一个episode不进行保存，而直接在后面使用torch.save进行保存
            if ((iteration + 1) % self.args.save_freq == 0) and (iteration + 1) != total_iterations:
                self.save_checkpoint(iteration + 1)

            # 这个是说test_iters个episode之后进行一次测试，有就是用正在训练的模型进行一次测试
            if ((iteration + 1) in self.args.test_iters) and (iteration + 1) != total_iterations:
                accuracy_dict = self.test(iteration + 1)
                print(accuracy_dict)
                self.test_accuracies.print(self.logfile, accuracy_dict)

    def train_task(self, task_dict, iteration):
        # 输入是task_dict 由video_reader读取得到的对象， 输出 support样本, query样本, support labels, query labels,  real_target_labels
        context_images, target_images, context_labels, target_labels, real_support_labels, real_target_labels, batch_class_list = self.prepare_task(task_dict)
        # 跑模型
        for k in task_dict.keys():
            task_dict[k] = task_dict[k][0].cuda()
        model_dict = self.model(task_dict, iteration)

        class_logits = model_dict['class_logits']
        meta_logits = model_dict['meta_logits']

        meta_kl_loss = model_dict['meta_kl_loss']
        superviesed_kl_loss = model_dict['superviesed_kl_loss']
        reconstruct_distance_loss = model_dict['reconstruct_distance']

        #元学习精确度
        meta_logits_total = meta_logits
        task_accuracy = self.accuracy_fn(meta_logits_total, target_labels)
        print("-> meta task_accuracy:{}".format(task_accuracy))
        #每个eipsode中监督学习准确度

        pre_task_accuracy = self.accuracy_fn(class_logits, torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]]))
        print("-> pre train train_accuracy:{}".format(pre_task_accuracy))


        class_logits = class_logits.unsqueeze(0)
        superviesed_class_loss = self.loss(class_logits, torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long(), "cuda") / self.args.tasks_per_batch
        meta_target_logits = meta_logits.unsqueeze(0)
        meta_target_loss = self.loss(meta_target_logits, task_dict["target_labels"].long(), "cuda") / self.args.tasks_per_batch

        #center_loss = model_dict['center_loss']
        #center_loss = 0
        lambdas = self.args.lambdas
        #lambdas=[2, 1, 0.5, 0.3, 0.3, 0.5, 0.5,0]start_cross
        lambdas = [2, 1, 0.5, 0, 0, 0]
        if iteration > self.args.start_cross:
            lambdas = [0, 1, 0, 0.5, 0.1]
            lambdas = [1, 0.2, 0.2, 0.1, 0.1, 0.1]
            lambdas = [1, 0.2, 0.2, 0, 0]
            #8种组合
            #lambdas = [1, 0, 0.2, 0, 0]    #single supervised 无蒸馏
            #lambdas = [0, 1, 0.2, 0, 0]     #single meta       无蒸馏
            #lambdas = [1, 1, 0.2, 0, 0]    #supervied + meta  无蒸馏

            #lambdas = [1, 0, 0.2, 0.2, 0]  #supervied + supervised     蒸馏
            #lambdas = [0, 1, 0.2, 0, 0.2]  #meta + meta                蒸馏

            #lambdas = [1, 1, 0.2, 0.2, 0]  #supervied+meta + supervised蒸馏
            #lambdas = [1, 1, 0.2, 0, 0.2]  #upervied+meta + meta       蒸馏
            lambdas = [1, 1, 0, 0.2, 0.2]  #upervied+meta + meta 和 supervised蒸馏

            #lambdas = [1, 1, 0, 0.2, 0.2] # 在meta阶段不要一致性损失

        else:
            lambdas = [1, 0, 1, 0, 0, 0]

        task_loss_total = lambdas[0] * superviesed_class_loss + \
                          lambdas[1] * meta_target_loss + \
                          lambdas[2] * reconstruct_distance_loss + \
                          lambdas[3] * superviesed_kl_loss + \
                          lambdas[4] * meta_kl_loss


        task_loss_total.backward(retain_graph=False)

        return task_loss_total, task_accuracy

    def test(self, num_episode):
        self.model.eval()  # 在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移。
        self.video_loader.dataset.train = False # 此参数为False,则video_loader加载的是测试数据集
        ####################################################
        with torch.no_grad():
            accuracy_dict = {}
            accuracies = []
            losses = []
            iteration = 0

            tmp_accuracies = []
            tmp_losses = []
            # 数据集的名称
            item = self.args.target_dataset
            for task_dict in self.video_loader:
                if iteration >= self.args.num_test_tasks: #num_test_tasks为测试中总的episode数量
                    break
                iteration += 1
                task_accuracy, task_loss_total = self.test_task(task_dict)
                eval_logger.info("For Task: {0}, the testing loss is {1} and Testing Accuracy is {2}".format(iteration + 1,task_loss_total, task_accuracy.item()))
                print("For Task: {0}, the testing loss is {1} and Testing Accuracy is {2}".format(iteration + 1, task_loss_total, task_accuracy.item()))
                losses.append(task_loss_total.item())
                accuracies.append(task_accuracy.item())
                ###########################################################################################################
                '''以下代码块的内容是为了进行一个较小集合的计算 作为参考'''
                tmp_losses.append(task_loss_total.item())
                tmp_accuracies.append(task_accuracy.item())
                if (iteration - 1) % 100 == 0 and iteration > 1:
                    tmp_accuracy = np.array(tmp_accuracies).mean() * 100.0
                    # 总体标准差 等于 样本标准差除以根号下样本数量n
                    tmp_confidence = (196.0 * np.array(tmp_accuracies).std()) / np.sqrt(len(tmp_accuracies))
                    tmp_loss = np.array(losses).mean()  # loss取均值
                    accuracy_dict[item] = {"accuracy": tmp_accuracy, "confidence": tmp_confidence, "loss": tmp_loss}
                    print("#####################   The databse is {}, and the iteration is {}   ######################".format(item, num_episode))
                    print(accuracy_dict)
                    print("##############################################################################################")
                    tmp_accuracies = []
                    tmp_losses = []
                #############################################################################################################
            accuracy = np.array(accuracies).mean() * 100.0
            # 总体标准差 等于 样本标准差除以根号下样本数量n
            confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
            loss = np.array(losses).mean()  # loss取均值
            accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence, "loss": loss}
            eval_logger.info("#####################   The databse is {}, and the iteration is {}   ######################".format(item, num_episode))
            eval_logger.info(accuracy_dict)
            eval_logger.info("##############################################################################################")
            eval_logger.info("----------------------------------------------------------------------------------------------")
            # send_email("hhhizsdjpwqnbaif", "accuracy " + str(accuracy), "769038835@qq.com")
        ####################################################

        self.video_loader.dataset.train = True
        self.model.train()

        return accuracy_dict

    def test_task(self, task_dict):
        context_images, target_images, context_labels, target_labels, real_support_labels, real_target_labels, batch_class_list = self.prepare_task(task_dict)

        for k in task_dict.keys():
            if  len(task_dict[k]) > 0:
                task_dict[k] = task_dict[k][0].cuda()
        model_dict = self.model.forward_test(task_dict)

        target_logits = model_dict['dis_logits']
        target_logits_total = target_logits
        task_accuracy = self.accuracy_fn(target_logits_total, target_labels)
        print("--> task_accuracy:{}".format(task_accuracy))

        target_logits = target_logits.unsqueeze(0)
        target_loss = self.loss(target_logits, task_dict["target_labels"].long(), "cuda") / self.args.tasks_per_batch
        lambdas = self.args.lambdas
        task_loss_total = lambdas[0] * target_loss
        return task_accuracy, task_loss_total

    def prepare_task(self, task_dict, images_to_device=True):
        context_images, context_labels = task_dict['support_set'][0], task_dict['support_labels'][0]
        target_images, target_labels = task_dict['target_set'][0], task_dict['target_labels'][0]
        real_support_labels = task_dict['real_support_labels'][0]
        real_target_labels = task_dict['real_target_labels'][0]
        batch_class_list = task_dict['batch_class_list'][0]

        if images_to_device:
            context_images = context_images.cuda()
            target_images = target_images.cuda()
        context_labels = context_labels.cuda()
        target_labels = target_labels.type(torch.LongTensor).cuda()

        return context_images, target_images, context_labels, target_labels, real_support_labels, real_target_labels, batch_class_list

    # 此函數無用。
    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]

    def save_checkpoint(self, iteration):
        d = {'iteration': iteration,
             'model_state_dict': self.model.state_dict(),
             'optimizer_state_dict': self.optimizer.state_dict(),
             'scheduler': self.scheduler.state_dict()}

        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint{}.pt'.format(iteration)))
        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self, test_checkpoint_iter):
        if test_checkpoint_iter:
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'checkpoint{}.pt'.format(test_checkpoint_iter)))
        else:
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'checkpoint.pt'))
        self.start_iteration = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])


if __name__ == "__main__":
    main()
