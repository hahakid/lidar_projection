#import kitti_object as ko
#import kitti_util as utils
import os
import sys
import numpy as np
import cv2, math
#cbox = np.array([[0, 80], [-40, 40], [-2, 1.25]])
BVF=True
# boundary
TOP_Y_MIN = -30
TOP_Y_MAX = +30
TOP_X_MIN = 0
TOP_X_MAX = 60
TOP_Z_MIN = -2
TOP_Z_MAX = 1.2
# final pixel= [(ymax-ymin)/y_d]*[(xmax-xmin)/x_d]
TOP_X_DIVISION = 0.1# 1 pixel=0.1 m
TOP_Y_DIVISION = 0.1
TOP_Z_DIVISION = 0.2

#main path
root = r'H:\kitti\tracking'
data_path = os.path.join(root, 'training')

#removed class
no_care=['DontCare','Pedestrian','Cyclist','Misc']

#label structure
class Object3d(object):
    """ 3d object label for tracking"""
    def __init__(self, label_file_line):
        data = label_file_line.split(" ")
        self.trackID=data[1]#0-frameID 1-trackID
        data[3:] = [float(x) for x in data[3:]]
        # extract label, truncation, occlusion
        self.type = data[2]  # 'Car', 'Pedestrian', ...
        self.truncation = data[3]  # truncated pixel ratio [0..1]
        self.occlusion = int(
            data[4]
        )  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[5]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[6]  # left
        self.ymin = data[7]  # top
        self.xmax = data[8]  # right
        self.ymax = data[9]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[10]  # box height
        self.w = data[11]  # box width
        self.l = data[12]  # box length (in meters)
        self.t = (data[13], data[14], data[15])  # location (x,y,z) in camera coord.
        self.ry = data[16]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def getID(self):
        return self.trackID

def inverse_rigid_trans(Tr):
    """ Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr



def read_label(lines):
    #lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects

#Calib structure & functions
class Calibration(object):
    """ Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    """

    def __init__(self, calib_filepath, from_video=False):

        calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs["P2"]# cam 2 left color
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs["Tr_velo_cam"] # different from object
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs["R_rect"] # different from object
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        """ Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def project_rect_to_velo(self, pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        """
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_rect_to_ref(self, pts_3d_rect):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))
    def cart2hom(self, pts_3d):
        """ Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom


#llist is a
def trans_label(llist):
    cl=[]
    for l in llist:
        it=l.strip('\n').split(' ')
        cls=it[2]
        truncation=it[3]
        occlusion=it[4]
        alpha=float(it[5])
        x1=float(it[6])
        y1=float(it[7])
        x2=float(it[8])
        y2=float(it[9])

        h=float(it[10])
        w=float(it[11])
        l=float(it[12])
        x=float(it[13])
        y=float(it[14])
        z=float(it[15])
        yaw=float(it[16])
        cc=[cls,h,w,l,x,y,z,yaw]
        cl.append(cc)
    return cl

def lidar_to_top(lidar):

    mask = np.where((lidar[:, 0] > TOP_X_MIN) & (lidar[:, 0] < TOP_X_MAX) & (lidar[:, 1] > TOP_Y_MIN) & (
            lidar[:, 1] < TOP_Y_MAX) & (lidar[:, 2] > TOP_Z_MIN) & (lidar[:, 2] < TOP_Z_MAX))
    lidar = lidar[mask]
    #lidar[:, 2] = lidar[:, 2] + TOP_Z_MIN  # lidar plane--> vehicle plane
    '''
    idx = np.where(lidar[:, 0] > TOP_X_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 0] < TOP_X_MAX)
    lidar = lidar[idx]

    idx = np.where(lidar[:, 1] > TOP_Y_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 1] < TOP_Y_MAX)
    lidar = lidar[idx]

    idx = np.where(lidar[:, 2] > TOP_Z_MIN)
    lidar = lidar[idx]
    idx = np.where(lidar[:, 2] < TOP_Z_MAX)
    lidar = lidar[idx]
    '''

    pxs = lidar[:, 0]# x
    pys = lidar[:, 1]# y
    pzs = lidar[:, 2]# z
    prs = lidar[:, 3]# reflection

    #trans to (0,0,0)  //整除
    qxs = ((pxs - TOP_X_MIN) // TOP_X_DIVISION).astype(np.int32)
    qys = ((pys - TOP_Y_MIN) // TOP_Y_DIVISION).astype(np.int32)
    # qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)
    qzs = (pzs - TOP_Z_MIN) / TOP_Z_DIVISION # lidar plane--> vehicle plane

    quantized = np.dstack((qxs, qys, qzs, prs)).squeeze()# 1,n,4 => n*4

    X0, Xn = 0, int((TOP_X_MAX - TOP_X_MIN) // TOP_X_DIVISION) + 1
    Y0, Yn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_DIVISION) + 1
    Z0, Zn = 0, int((TOP_Z_MAX - TOP_Z_MIN) / TOP_Z_DIVISION)
    height = Xn - X0
    width = Yn - Y0
    channel = Zn - Z0 + 2

    # print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height, width, channel), dtype=np.float32)

    # histogram = Bin(channel, 0, Zn, "z", Bin(height, 0, Yn, "y", Bin(width, 0, Xn, "x", Maximize("intensity"))))
    # histogram.fill.numpy({"x": qxs, "y": qys, "z": qzs, "intensity": prs})

    if 1:  # new method
        for x in range(Xn):
            ix = np.where(quantized[:, 0] == x)
            quantized_x = quantized[ix]
            if len(quantized_x) == 0:
                continue
            yy = -x

            for y in range(Yn):
                iy = np.where(quantized_x[:, 1] == y)
                quantized_xy = quantized_x[iy]
                count = len(quantized_xy)
                if count == 0:
                    continue
                xx = -y

                top[yy, xx, Zn + 1] = min(1, np.log(count + 1) / math.log(32))
                max_height_point = np.argmax(quantized_xy[:, 2])
                top[yy, xx, Zn] = quantized_xy[max_height_point, 3]

                for z in range(Zn):
                    iz = np.where(
                        (quantized_xy[:, 2] >= z) & (quantized_xy[:, 2] <= z + 1)
                    )
                    quantized_xyz = quantized_xy[iz]
                    if len(quantized_xyz) == 0:
                        continue
                    zz = z
                    # height per slice
                    max_height = max(0, np.max(quantized_xyz[:, 2]) - z)
                    top[yy, xx, zz] = max_height
    return top

def makeBVFeature(lidar):
    mask = np.where((lidar[:, 0] > TOP_X_MIN) & (lidar[:, 0] < TOP_X_MAX) & (lidar[:, 1] > TOP_Y_MIN) & (
            lidar[:, 1] < TOP_Y_MAX) & (lidar[:, 2] > TOP_Z_MIN) & (lidar[:, 2] < TOP_Z_MAX))
    lidar = lidar[mask]
    pxs = lidar[:, 0]  # x
    pys = lidar[:, 1]  # y
    pzs = lidar[:, 2]  # z
    prs = lidar[:, 3]  # reflection
    qxs = ((pxs - TOP_X_MIN) // TOP_X_DIVISION).astype(np.int32)
    qys = ((pys - TOP_Y_MIN) // TOP_Y_DIVISION).astype(np.int32)
    # qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)
    qzs = (pzs - TOP_Z_MIN) / TOP_Z_DIVISION # lidar plane--> vehicle plane

    quantized = np.dstack((qxs, qys, qzs, prs)).squeeze()# 1,n,4 => n*4
    X0, Xn = 0, int((TOP_X_MAX - TOP_X_MIN) // TOP_X_DIVISION) + 1
    Y0, Yn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_DIVISION) + 1
    Z0, Zn = 0, int((TOP_Z_MAX - TOP_Z_MIN) / TOP_Z_DIVISION)
    height = Xn - X0
    width = Yn - Y0
    zind=Zn-Z0+2

    top = np.zeros(shape=(height, width, 3), dtype=np.float32)# reflection height count

    #
    heightMap = np.zeros(shape=(height, width), dtype=np.float32)
    reflectionMap = np.zeros(shape=(height, width), dtype=np.float32)
    countMap = np.zeros(shape=(height, width), dtype=np.float32)

    indices = np.lexsort((quantized[:, 0], quantized[:, 1], quantized[:, 2]))
    PointCloud = quantized[indices]
    _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
    PointCloud_frac = PointCloud[indices]

    heightMap[np.int_(-PointCloud_frac[:, 0]), -np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 2]
    reflectionMap[np.int_(-PointCloud_frac[:, 0]), -np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 3]

    _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[indices]
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    countMap[np.int_(-PointCloud_top[:, 0]), -np.int_(PointCloud_top[:, 1])] = normalizedCounts

    heightMap=np.dot(heightMap/heightMap.max(),255)
    reflectionMap = np.dot(reflectionMap / reflectionMap.max(), 255)
    countMap = np.dot(countMap / countMap.max(), 255)

    #cv2.imshow("0",heightMap)
    #cv2.imshow("1", reflectionMap)
    #cv2.imshow("2", countMap)
    #cv2.waitKey()

    top[:, :, 0] =  heightMap # r_map
    top[:, :, 1] = reflectionMap  # g_map
    top[:, :, 2] = countMap  # b_map
    return top


def draw_top_image(lidar_top):
    top_image = np.sum(lidar_top, axis=2)
    top_image = top_image - np.min(top_image)
    divisor = np.max(top_image) - np.min(top_image)
    top_image = top_image / divisor * 255
    top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)
    return top_image

def lidar_to_top_coords(x, y, z=None):
    if 0:
        return x, y
    else:
        # print("TOP_X_MAX-TOP_X_MIN:",TOP_X_MAX,TOP_X_MIN)
        X0, Xn = 0, int((TOP_X_MAX - TOP_X_MIN) // TOP_X_DIVISION) + 1
        Y0, Yn = 0, int((TOP_Y_MAX - TOP_Y_MIN) // TOP_Y_DIVISION) + 1
        xx = Yn - int((y - TOP_Y_MIN) // TOP_Y_DIVISION)
        yy = Xn - int((x - TOP_X_MIN) // TOP_X_DIVISION)
        return xx, yy

def heat_map_rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return (r, g, b)

def draw_box3d_on_top(
    image,
    boxes3d,
    color=(255, 255, 255),
    thickness=1,
    scores=None,
    text_lables=[],
    is_gt=False,
):

    # if scores is not None and scores.shape[0] >0:
    # print(scores.shape)
    # scores=scores[:,0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = image.copy()
    num = len(boxes3d)
    startx = 5
    fin_label=[]#输出label
    for n in range(num):
        b = boxes3d[n]
        x0 = b[0, 0]
        y0 = b[0, 1]
        x1 = b[1, 0]
        y1 = b[1, 1]
        x2 = b[2, 0]
        y2 = b[2, 1]
        x3 = b[3, 0]
        y3 = b[3, 1]
        u0, v0 = lidar_to_top_coords(x0, y0)
        u1, v1 = lidar_to_top_coords(x1, y1)
        u2, v2 = lidar_to_top_coords(x2, y2)
        u3, v3 = lidar_to_top_coords(x3, y3)
        if is_gt:
            color = (0, 255, 0)
            startx = 5
        else:
            color = heat_map_rgb(0.0, 1.0, scores[n]) if scores is not None else 255
            startx = 85
        cv2.line(img, (u0, v0), (u1, v1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u1, v1), (u2, v2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u2, v2), (u3, v3), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u3, v3), (u0, v0), color, thickness, cv2.LINE_AA)
        fin_label.append(text_lables[n][0]+" "+text_lables[n][1]+" "+str(u0)+" "+str(v0)+" "+str(u1)+" "+str(v1)+" "+str(u2)+" "+str(v2)+" "+str(u3)+" "+str(v3))
    for n in range(len(text_lables)):
        text_pos = (startx, 25 * (n + 1))
        cv2.putText(img, text_lables[n][0]+'(ID:'+text_lables[n][1]+')', text_pos, font, 0.5, color, 0, cv2.LINE_AA)

    return img,fin_label

def project_to_image(pts_3d, P):
    """ Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    """
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]

def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def compute_box_3d(obj, P):
    """ Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)

def show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred=None,is_show=False):
    """ top_view image"""
    # print('pc_velo shape: ',pc_velo.shape)

    if BVF==True:
        top_image = makeBVFeature(pc_velo)
    else:
        top_view = lidar_to_top(pc_velo)
        top_image = draw_top_image(top_view)
    #print("top_image:", top_image.shape)
    # gt
    outimg = top_image
    def bbox3d(obj):
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        return box3d_pts_3d_velo

    boxes3d = [bbox3d(obj) for obj in objects if obj.type != "DontCare"]
    gt = np.array(boxes3d)
    # print("box2d BV:",boxes3d)
    lines = [[obj.type,obj.trackID] for obj in objects if obj.type != "DontCare"]
    IDs=[obj.trackID for obj in objects if obj.type != -1]
    top_image,pro_labels = draw_box3d_on_top(
        top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=True
    )
    # pred
    if objects_pred is not None:
        boxes3d = [bbox3d(obj) for obj in objects_pred if obj.type != "DontCare"]
        gt = np.array(boxes3d)
        lines = [obj.type for obj in objects_pred if obj.type != "DontCare"]
        top_image = draw_box3d_on_top(
            top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=False
        )
    if is_show:
        cv2.imshow("top_image", top_image)
        cv2.waitKey(30)
    return top_image,outimg,pro_labels

#seq control
for seq in range(11,12):

    lidar_path = os.path.join(data_path, "velodyne/",str(seq).zfill(4))
    image_path = os.path.join(data_path, "image_02/",str(seq).zfill(4))
    calib_path = os.path.join(data_path, "calib/",str(seq).zfill(4)+'.txt')
    label_path = os.path.join(data_path, "label_02/",str(seq).zfill(4)+'.txt')
    birdview_path=os.path.join(data_path,'birdview/',str(seq).zfill(4)+'/')

    f=open(label_path,'r')
    all_labels=f.readlines()
    f.close()

    if not os.path.exists(birdview_path):
        os.makedirs(birdview_path)
    #calib =utils.read_calib_file(calib_path)
    calib=Calibration(calib_path) # one for sequence

    vout = cv2.VideoWriter()
    fps = 10
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    size=(int((TOP_X_MAX-TOP_X_MIN)/TOP_X_DIVISION),int((TOP_Y_MAX-TOP_Y_MIN)/TOP_Y_DIVISION))
    vout=cv2.VideoWriter(birdview_path+str(seq).zfill(4)+'.mp4', fourcc, fps, size)

    for i in range(0,len(os.listdir(lidar_path))):#item frame
        cur_labels=[]# label for current frame
        for l in all_labels:
            if l.split(' ')[0]== str(i) and l.split(' ')[2] not in no_care:
                cur_labels.append(l)
        #cl=trans_label(cur_labels)
        #for ccl in cl:
        #    print(ccl)
        objects = read_label(cur_labels)#get label
        #n_obj =len(objects)

        lidar_file = lidar_path + "/" + str(i).zfill(6) + '.bin'
        pc_velo=np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        lab_img,fin_img,fin_label=show_lidar_topview_with_boxes(pc_velo, objects, calib,is_show=False)

        cv2.imwrite(os.path.join(birdview_path,str(i).zfill(6)+'.png'),fin_img)
        f_label=open(os.path.join(birdview_path,str(i).zfill(6)+'.txt'),'a')
        lab_img=np.uint8(lab_img)
        #print(lab_img.shape)
        vout.write(lab_img)
        for f in fin_label:
            f_label.write(f+'\n')
        f_label.close()
    vout.release()
    cv2.destroyAllWindows()




