
6
import sys
import cv2
import numpy as np
import math
import networkx as nx
import os
import matplotlib.pyplot as plt
import re
from glob import glob
from slic_segmentation import *

from skimage.util import img_as_float
from skimage.color import rgb2gray
from skimage.color import gray2rgb
from skimage.color import rgb2lab
from skimage.segmentation import slic
import scipy.spatial.distance

# path = './VOCROSD/JPEGImages'
# path_1 = './VOCROSD/JPEGImages1'

save_path ="./VOCROSD/JPEGImages4"# 保存文件的路径
path = "G:/1.course/myproject/3saliency/faster_rcnngai/VOCdevkit/DIOR/JPEGImages-test" # 原文件的路径G:\1.course\myproject\3saliency\faster_rcnngai\VOCdevkit\DIOR
frames = glob(os.path.join(path, '*.jpg'))


def S(x1,x2,geodesic,sigma_clr=10):
    return math.exp(-pow(geodesic[x1,x2],2)/(2*sigma_clr*sigma_clr))

def compute_saliency_cost(smoothness,w_bg,wCtr):
    n = len(w_bg)
    A = np.zeros((n,n))
    b = np.zeros((n))

    for x in range(0,n):
        A[x,x] = 2 * w_bg[x] + 2 * (wCtr[x])
        b[x] = 2 * wCtr[x]
        for y in range(0,n):
            A[x,x] += 2 * smoothness[x,y]
            A[x,y] -= 2 * smoothness[x,y]

    x = np.linalg.solve(A, b)

    return x

def path_length(path,G):
    dist = 0.0
    for i in range(1,len(path)):
        dist += G[path[i - 1]][path[i]]['weight']
    return dist

def make_graph(grid):
    # get unique labels
    vertices = np.unique(grid)

    # map unique labels to [1,...,num_labels]
    reverse_dict = dict(zip(vertices,np.arange(len(vertices))))
    grid = np.array([reverse_dict[x] for x in grid.flat]).reshape(grid.shape)

    # create edges
    down = np.c_[grid[:-1, :].ravel(), grid[1:, :].ravel()]
    right = np.c_[grid[:, :-1].ravel(), grid[:, 1:].ravel()]
    all_edges = np.vstack([right, down])
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
    all_edges = np.sort(all_edges,axis=1)
    num_vertices = len(vertices)
    edge_hash = all_edges[:,0] + num_vertices * all_edges[:, 1]
    # find unique connections
    edges = np.unique(edge_hash)
    # undo hashing
    edges = [[vertices[x%num_vertices],
              vertices[x//num_vertices]] for x in edges]

    return vertices, edges


def get_saliency_rbd(img_path):

    # Saliency map calculation based on:
    # Saliency Optimization from Robust Background Detection, Wangjiang Zhu, Shuang Liang, Yichen Wei and Jian Sun, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014

    img = cv2.imread(img_path)

    if len(img.shape) != 3: # got a grayscale image
        img = gray2rgb(img)

    img_lab = img_as_float(rgb2lab(img))

    img_rgb = img_as_float(img)

    img_gray = img_as_float(rgb2gray(img))
    # p=SLIC(img_rgb,250,10);
    # segments_slic = p.iterate_times(5)
    segments_slic=slic(img_rgb, n_segments=300, compactness=10, sigma=0.7, enforce_connectivity=False)
    # print 'hello'
    num_segments = len(np.unique(segments_slic))

    nrows, ncols = segments_slic.shape
    max_dist = math.sqrt(nrows*nrows + ncols*ncols)

    grid = segments_slic

    (vertices,edges) = make_graph(grid)

    gridx, gridy = np.mgrid[:grid.shape[0], :grid.shape[1]]

    centers = dict()
    colors = dict()
    distances = dict()
    boundary = dict()

    for v in vertices:
        centers[v] = [gridy[grid == v].mean(), gridx[grid == v].mean()]
        colors[v] = np.mean(img_lab[grid==v],axis=0)

        x_pix = gridx[grid == v]
        y_pix = gridy[grid == v]

        if np.any(x_pix == 0) or np.any(y_pix == 0) or np.any(x_pix == nrows - 1) or np.any(y_pix == ncols - 1):
            boundary[v] = 1
        else:
            boundary[v] = 0

    G = nx.Graph()

    #buid the graph
    for edge in edges:
        pt1 = edge[0]
        pt2 = edge[1]
        color_distance = scipy.spatial.distance.euclidean(colors[pt1],colors[pt2])
        G.add_edge(pt1, pt2, weight=color_distance )

    #add a new edge in graph if edges are both on boundary
    for v1 in vertices:
        if boundary[v1] == 1:
            for v2 in vertices:
                if boundary[v2] == 1:
                    color_distance = scipy.spatial.distance.euclidean(colors[v1],colors[v2])
                    G.add_edge(v1,v2,weight=color_distance)

    geodesic = np.zeros((len(vertices),len(vertices)),dtype=float)
    spatial = np.zeros((len(vertices),len(vertices)),dtype=float)
    smoothness = np.zeros((len(vertices),len(vertices)),dtype=float)
    adjacency = np.zeros((len(vertices),len(vertices)),dtype=float)

    sigma_clr = 10.0
    sigma_bndcon = 1.0
    sigma_spa = 0.25
    mu = 0.1

    all_shortest_paths_color = nx.shortest_path(G,source=None,target=None,weight='weight')

    for v1 in vertices:
        for v2 in vertices:
            if v1 == v2:
                geodesic[v1,v2] = 0
                spatial[v1,v2] = 0
                smoothness[v1,v2] = 0
            else:
                geodesic[v1,v2] = path_length(all_shortest_paths_color[v1][v2],G)
                spatial[v1,v2] = scipy.spatial.distance.euclidean(centers[v1],centers[v2]) / max_dist
                smoothness[v1,v2] = math.exp( - (geodesic[v1,v2] * geodesic[v1,v2])/(2.0*sigma_clr*sigma_clr)) + mu

    for edge in edges:
        pt1 = edge[0]
        pt2 = edge[1]
        adjacency[pt1,pt2] = 1
        adjacency[pt2,pt1] = 1

    for v1 in vertices:
        for v2 in vertices:
            smoothness[v1,v2] = adjacency[v1,v2] * smoothness[v1,v2]

    area = dict()
    len_bnd = dict()
    bnd_con = dict()
    w_bg = dict()
    ctr = dict()
    wCtr = dict()

    for v1 in vertices:
        area[v1] = 0
        len_bnd[v1] = 0
        ctr[v1] = 0
        for v2 in vertices:
            d_app = geodesic[v1,v2]
            d_spa = spatial[v1,v2]
            w_spa = math.exp(- ((d_spa)*(d_spa))/(2.0*sigma_spa*sigma_spa))
            area_i = S(v1,v2,geodesic)
            area[v1] += area_i
            len_bnd[v1] += area_i * boundary[v2]
            ctr[v1] += d_app * w_spa
        bnd_con[v1] = len_bnd[v1] / math.sqrt(area[v1])
        w_bg[v1] = 1.0 - math.exp(- (bnd_con[v1]*bnd_con[v1])/(2*sigma_bndcon*sigma_bndcon))

    for v1 in vertices:
        wCtr[v1] = 0
        for v2 in vertices:
            d_app = geodesic[v1,v2]
            d_spa = spatial[v1,v2]
            w_spa = math.exp(- (d_spa*d_spa)/(2.0*sigma_spa*sigma_spa))
            wCtr[v1] += d_app * w_spa *  w_bg[v2]

    # normalise value for wCtr

    min_value = min(wCtr.values())
    max_value = max(wCtr.values())

    minVal = [key for key, value in wCtr.items() if value == min_value]
    maxVal = [key for key, value in wCtr.items() if value == max_value]

    for v in vertices:
        wCtr[v] = (wCtr[v] - min_value)/(max_value - min_value)

    img_disp1 = img_gray.copy()
    img_disp2 = img_gray.copy()

    x = compute_saliency_cost(smoothness,w_bg,wCtr)

    for v in vertices:
        img_disp1[grid == v] = x[v]

    img_disp2 = img_disp1.copy()
    sal = np.zeros((img_disp1.shape[0],img_disp1.shape[1],3))

    sal = img_disp2
    sal_max = np.max(sal)
    sal_min = np.min(sal)
    sal = 255 * ((sal - sal_min) / (sal_max - sal_min))

    return sal

def save_saliency_rbd(img):

    # img = cv2.imread(img_path)
    image_np = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return image_np
if __name__ == '__main__':
    # 批量操作
    # for i, frame in enumerate(frames):
    #     pattern = re.compile(r'([^<>/\\\|:""\*\?]+)\.\w+$')
    #     data = pattern.findall(frame)
    #     data_now = data[0]
    #     Newdir = os.path.join(path, str(data_now) + '.jpg')
    #     img = cv2.imread(Newdir)
    #     cv2.waitKey(500)
    #
    #     rbd = get_saliency_rbd(frame).astype('uint8')

        # # image_np = save_saliency_rbd(rbd)
        # # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #
        # name = str(data_now) + ".jpg"
        # cv2.imwrite(save_path + '/' + name, rbd)
        # print(name)
        # filename = "./aircraft_32.jpg"
        # rbd = get_saliency_rbd(filename).astype('uint8')
        #
        # adaptive_threshold = 2.0 * rbd.mean()
        # final_binary = (rbd > adaptive_threshold)
        #
        # img = cv2.imread(filename)
        #
        # cv2.waitKey(500)
        #
        # cv2.imshow('img', rbd)
        # # print(rbd)
        # cv2.imwrite('Saliency2_' + filename, rbd)
        # cv2.imshow('img1', 255 * final_binary.astype('uint8'))
        # cv2.imwrite('Mask2_' + filename, 255 * final_binary.astype('uint8'))
        # # cv2.imwrite('Saliency2_'+filename,rbd)
        # cv2.imshow('Mask2_'+filename,255 * final_binary.astype('uint8'))
        #
        # cv2.imshow('binary', 255 * final_binary.astype('uint8'))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
#     for info in os.listdir(path):
#         domain = os.path.abspath(path)
#         info1 = os.path.join(domain,info)
#         print(info1)
#
#         img = cv2.imread(info1)
#         cv2.imshow('original',img)
#         cv2.waitKey(1000)
#         rbd = get_saliency_rbd(info1).astype('uint8')
#         image_np=save_saliency_rbd(rbd)
#         cv2.imshow('Flipped Horizontalliy-Verticalliy', image_np)
#         cv2.imwrite(save_path + str(i) + '.jpg', image_np)
#
        # if(len(argv)!=2):
        #     print( 'python Saliency_Optimization #ImageName')
        #     filename=sys.argv[1];
        filename = "./00975.jpg"
        # filename = "./overpass_250.jpg"
        rbd = get_saliency_rbd(filename).astype('uint8')
        #
        adaptive_threshold = 2.0 * rbd.mean()
        final_binary = (rbd > adaptive_threshold)

        img = cv2.imread(filename)
        cv2.imshow('img',rbd)
        cv2.imwrite('Saliency2_'+filename,rbd)
        cv2.imwrite('Mask2_'+filename,255 * final_binary.astype('uint8'))

        cv2.imshow('binary',255 * final_binary.astype('uint8'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        image_np = cv2.cvtColor(rbd, cv2.COLOR_GRAY2BGR)
        fig = plt.figure("gray_image")
        ax = fig.add_subplot(1,1,1)
        # cv2.imshow('img', rbd)
        ax.imshow(rbd)
        # plt.imshow(rbd)
        # print(ax)
        # cv2.imwrite('./Op' + '.jpg', rbd)
        # ax.imshow(rbd)

        plt.axis("off")
        plt.savefig('./six.png', dpi=1000, bbox_inches='tight', pad_inches=0)
        plt.show()

    #
    # cv2.imwrite('./pass2' + '.jpg', image_np)
    # # plt.savefig('./1.png', rbd)
    # # plt.savefig
#
# # import os
# # save_path = "./post/" # 保存文件的路径
# # path = "./src/"  # 原文件的路径
# # j = 0
# # for info in os.listdir(path):
# #     domain = os.path.abspath(path)
# #     info1 = os.path.join(domain,info)
# #     img = cv2.imread(info1)
# #     cv2.imshow('original',img)
# #     cv2.waitKey(1000)
# #
# #     # 水平镜像
# #     i = j
# #     hv_flip = cv2.flip(img, flipCode=-1)
# #     cv2.imshow('Flipped Horizontalliy-Verticalliy',hv_flip)
# #     cv2.imwrite(save_path + str(i) + '.jpg',hv_flip)
# #
# #     # 缩放 放大
# #     i = i+1
# #     height, width = img.shape[:2]
# #     res = cv2.resize(img,(2*width,2*height))
# #     cv2.imshow("large",res)
# #     cv2.imwrite(save_path + str(i) + '.jpg', res)
# #     j = i+1
# #
# # cv2.waitKey(0)
# # dir = "F:\\image\\save\\"
# # video_path = "F:\image"
# # frames = glob(os.path.join(video_path, '*.jpg'))
# #
# # for i, frame in enumerate(frames):
# #     pattern = re.compile(r'([^<>/\\\|:""\*\?]+)\.\w+$')
# #     data = pattern.findall(frame)
# #     data_now = data[0]
# #     Newdir = os.path.join(video_path, str(data_now) + '.jpg')
# #     img = cv2.imread(Newdir)
# #
# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #
# #     name = str(data_now) + ".jpg"
# #     cv2.imwrite(dir + name, img)
