import config as cfg
import os
import pandas as pd
import numpy as np
from v3dpy.loaders import Raw, PBD
from published_output import complete_tiff, nonempty_file, publish_derived_file
from sample_preparation_paths import resolve_crop_metadata_root

import os
import pandas as pd
import numpy as np

class DataAccess:
    def __init__(self, soma_dir, img_dir):
        """
        :param soma_dir: soma标记文件路径顶层目录
        :param img_dir: 原始图像文件路径顶层目录
        """
        self.soma_dir = soma_dir
        self.img_dir = img_dir
    
    def get_soma_list(self, ptrsid, docid):
        """
        获取单个 document 的 soma 坐标信息
        :param ptrsid: Pxxxx-Txxx-Rxxx-Sxxx 等前缀 ID
        :param docid: 具体的 document_id (形如 P00125-T001-R002-S009-B1)
        :return: 包含 x, y, z 等信息的 DataFrame
        """
        apo_file = os.path.join(self.soma_dir, ptrsid, docid, docid + '_initial.apo')
        if not os.path.exists(apo_file):
            print(f"Can not find soma file: {apo_file}")
            return None
        
        somalist = pd.read_csv(apo_file)
        return somalist

    def get_image_data(self, image_path, raw_loader):
        """
        获取单个 document 的三维数据
        :return: np.array 形状 (c, z, y, x)
        """
        if not os.path.isfile(image_path):
            print(f"Can not find image file: {image_path}")
            return None
        
        img = raw_loader.load(image_path)
        return img


def fp2dbdirs(strcid):
    '''convert file path to Database dirs with dynamic directory name length'''
    cid = int(strcid)

    # 获取必要的数字长度（根据最大值确定）
    # first layer of dir: groups of 1000
    minb = int(cid / 1000) * 1000
    maxb = (int(cid / 1000) + 1) * 1000 - 1

    # 确定需要的位数
    digits_needed = max(len(str(maxb)), len(str(minb)))

    # 格式化目录名
    str_minb = str(minb).zfill(digits_needed)
    str_maxb = str(maxb).zfill(digits_needed)
    fdir = str_minb + '_' + str_maxb

    # second layer of dir: groups of 100
    layer_base = int(cid / 1000) * 1000
    minb1 = layer_base + int((cid - layer_base) / 100) * 100
    maxb1 = layer_base + int((cid - layer_base) / 100 + 1) * 100 - 1

    # 确定二级目录需要的位数
    digits_needed1 = max(len(str(maxb1)), len(str(minb1)))

    # 格式化二级目录名
    str_minb1 = str(minb1).zfill(digits_needed1)
    str_maxb1 = str(maxb1).zfill(digits_needed1)
    fdir1 = str_minb1 + '_' + str_maxb1

    todir2 = os.path.join(fdir, fdir1)
    return todir2


def main(h5_path):
    # h5_image_name (Input Parm 1)
    # Soma Marker Files: Located in /PBS/BRAINTELL/Projects/HumanNeurons/AllBrainSlices/HNDB_files/SamplePreparation/{ptrsid}/{docid}/{docid}_initial.apo
    # Raw Image Files: Located in /PBS/BRAINTELL/Projects/HumanNeurons/AllBrainSlices/PTRSB_DB/{ptrsid}/{docid}_8bit.v3draw

    # Cell Id Range (Input Parm 2)
    # cell_id_begin = 1
    # cell_id_end = 71659

    # {
    #     "h5_image_name": "PTRSB_DB",
    #     "cell_id_begin": 1,
    #     "cell_id_end": 71659
    # }

    # Begin Process #
    raw_loader = Raw()
    cell_pbd_saver = PBD()
 
    # 图像文件根目录
    img_dir = cfg.PTRSB_DBDirectory
    
    # 图像的裁剪结果保存目录
    topbd = cfg.Cell_ImagesDirectory
    
    xy_size = 700  # 裁剪图像边长的一半

    h5_image_name = os.path.basename(h5_path)
    if h5_image_name.endswith(".pyramid.h5"):
        docid = h5_image_name[:-len(".pyramid.h5")]
        image_path = h5_path[:-len(".pyramid.h5")] + "_8bit.v3draw"
    elif h5_image_name.endswith(".h5"):
        docid = h5_image_name[:-len(".h5")]
        image_path = h5_path[:-len(".h5")] + "_8bit.v3draw"
    else:
        raise ValueError("Expected an H5 input path")

    # 处理 docid，得到 ptrsid
    ptrsid = docid
    docid_split = docid.split('-')
    if len(docid_split) == 5: # P-T-R-S-B
        # 如果末位不是 Bxxx，则删去最后一个段
        if not docid_split[-1].startswith('B'):
            ptrsid = docid[0:-1 * (len(docid_split[4]) + 1)]
    elif len(docid_split) == 6: # P-T-R-S-B-N
        ptrsid = docid[0:-1 * (len(docid_split[5]) + 1)]

    soma_dir = resolve_crop_metadata_root(
        cfg.SamplePreparationDirectory,
        cfg.LegacySamplePreparationDirectory,
        ptrsid,
        docid,
    )
    data_accessor = DataAccess(soma_dir, img_dir)
        

    # 读取 CSV 文件获取 Cell ID 范围
    csv_file_path = os.path.join(soma_dir, ptrsid, docid, "cell_with_cellID.csv")
    if csv_file_path and os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path, encoding='UTF-8')
        cell_id_begin = int(df['Cell ID'].min())
        cell_id_end = int(df['Cell ID'].max())
        print(f"从CSV文件读取到的Cell ID范围: {cell_id_begin} - {cell_id_end}")
    else:
        raise Exception("未找到CSV文件")

    # 获取该 docid 最小的 cell_id，后续依次递增
    this_sid = cell_id_begin

    
    # 加载 soma 坐标
    somalist = data_accessor.get_soma_list(ptrsid, docid)
    if somalist is None:
        print("Soma info not found for:", docid)
        raise Exception("Soma info not found for: " + docid)
    
    print('Now:', docid, 'Somas=', somalist.shape[0])
    
    # 加载原始图像数据 (c, z, y, x)
    img = data_accessor.get_image_data(image_path, raw_loader)
    if img is None:
        print("Image data not found for:", docid)
        raise Exception("Image data not found for: " + docid)
    
    img_c, img_z, img_y, img_x = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    print(img_c, img_x, img_y, img_z)
    
    # 对每个 soma 坐标进行裁剪操作
    for s in somalist.index:
        sx = int(float(somalist.loc[s, 'x']))
        sy = int(float(somalist.loc[s, 'y']))
        sz = int(float(somalist.loc[s, 'z']))

        x_start = max(sx - xy_size, 0)
        x_end   = min(sx + xy_size + 1, img_x)
        y_start = max(sy - xy_size, 0)
        y_end   = min(sy + xy_size + 1, img_y)
        
        if x_start >= x_end or y_start >= y_end:
            raise ValueError(
                f"Invalid crop region for cell_id {this_sid} at x,y={sx},{sy}"
            )
        
        # 新的 soma 相对于裁剪子图的坐标
        newsx = sx - x_start
        newsy = sy - y_start
        
        # 保存结果的目录
        outsdir = os.path.join(topbd, fp2dbdirs(this_sid))
        if not os.path.exists(outsdir):
            print('Create directory:', outsdir)
            try:
                os.makedirs(outsdir, exist_ok=True)
                print(f'Successfully created directory: {outsdir}')
            except PermissionError:
                print(f'Permission denied: Cannot create directory {outsdir}')
                print('Please check if you have write permissions to this path')
                raise
            except Exception as e:
                print(f'Error creating directory {outsdir}: {e}')
                raise
        
        out_file = os.path.join(outsdir, str(this_sid) + '.v3dpbd')
        out_mip_dir = os.path.join(cfg.Cell_MIPsDirectory, fp2dbdirs(this_sid))
        os.makedirs(out_mip_dir, exist_ok=True)
        mip_file = os.path.join(out_mip_dir, str(this_sid) + '.tif')
        if not nonempty_file(out_file) or not complete_tiff(mip_file):
            # Compute once even if a previous run published only one output.
            cropped_img = img[:, :, y_start:y_end, x_start:x_end]
            publish_derived_file(
                out_file,
                lambda temporary: cell_pbd_saver.save(
                    temporary, np.array(cropped_img)
                ),
                nonempty_file,
            )
            from skimage import io

            publish_derived_file(
                mip_file,
                lambda temporary: io.imsave(
                    temporary,
                    np.max(cropped_img, axis=1).astype(np.uint8),
                ),
                complete_tiff,
            )
                

        # 递增 cell_id
        this_sid += 1
        
        # 检查保存是否成功
        if not nonempty_file(out_file) or not complete_tiff(mip_file):
            raise RuntimeError(f"Failed to publish crop outputs for cell {this_sid - 1}")
 

if __name__ == '__main__':
    # python3 {script_path} --image-path "{image_path}
    # 解析参数 --image-path
    import argparse
    parser = argparse.ArgumentParser(description="crop cell images from v3d files.")
    parser.add_argument('--image-path', type=str, required=True, help='Path to the v3d image file or directory.')
    args = parser.parse_args()

    input_path = args.image_path
    input_path = input_path.strip()

    main(input_path)
