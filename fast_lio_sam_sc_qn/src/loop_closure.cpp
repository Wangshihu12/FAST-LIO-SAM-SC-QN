#include "loop_closure.h"

LoopClosure::LoopClosure(const LoopClosureConfig &config)
{
    config_ = config;
    const auto &gc = config_.gicp_config_;
    const auto &qc = config_.quatro_config_;
    ////// nano_gicp init
    nano_gicp_.setNumThreads(gc.nano_thread_number_);
    nano_gicp_.setCorrespondenceRandomness(gc.nano_correspondences_number_);
    nano_gicp_.setMaximumIterations(gc.nano_max_iter_);
    nano_gicp_.setRANSACIterations(gc.nano_ransac_max_iter_);
    nano_gicp_.setMaxCorrespondenceDistance(gc.max_corr_dist_);
    nano_gicp_.setTransformationEpsilon(gc.transformation_epsilon_);
    nano_gicp_.setEuclideanFitnessEpsilon(gc.euclidean_fitness_epsilon_);
    nano_gicp_.setRANSACOutlierRejectionThreshold(gc.ransac_outlier_rejection_threshold_);
    ////// quatro init
    quatro_handler_ = std::make_shared<quatro<PointType>>(qc.fpfh_normal_radius_,
                                                          qc.fpfh_radius_,
                                                          qc.noise_bound_,
                                                          qc.rot_gnc_factor_,
                                                          qc.rot_cost_diff_thr_,
                                                          qc.quatro_max_iter_,
                                                          qc.estimat_scale_,
                                                          qc.use_optimized_matching_,
                                                          qc.quatro_distance_threshold_,
                                                          qc.quatro_max_num_corres_);
    src_cloud_.reset(new pcl::PointCloud<PointType>);
    dst_cloud_.reset(new pcl::PointCloud<PointType>);
}

LoopClosure::~LoopClosure() {}

void LoopClosure::updateScancontext(pcl::PointCloud<PointType> cloud)
{
    sc_manager_.makeAndSaveScancontextAndKeys(cloud);
}

int LoopClosure::fetchCandidateKeyframeIdx(const PosePcd &query_keyframe,
                                           const std::vector<PosePcd> &keyframes)
{
    // from ScanContext, get the loop candidate
    std::pair<int, float> sc_detected_ = sc_manager_.detectLoopClosureIDGivenScan(query_keyframe.pcd_); // int: nearest node index,
                                                                                                        // float: relative yaw
    int candidate_keyframe_idx = sc_detected_.first;
    if (candidate_keyframe_idx >= 0) // if exists
    {
        // if close enough
        if ((keyframes[candidate_keyframe_idx].pose_corrected_eig_.block<3, 1>(0, 3) - query_keyframe.pose_corrected_eig_.block<3, 1>(0, 3))
                .norm() < config_.scancontext_max_correspondence_distance_)
        {
            return candidate_keyframe_idx;
        }
    }
    return -1;
}

PcdPair LoopClosure::setSrcAndDstCloud(const std::vector<PosePcd> &keyframes,
                                       const int src_idx,
                                       const int dst_idx,
                                       const int submap_range,
                                       const double voxel_res,
                                       const bool enable_quatro,
                                       const bool enable_submap_matching)
{
    pcl::PointCloud<PointType> dst_accum, src_accum;
    int num_approx = keyframes[src_idx].pcd_.size() * 2 * submap_range;
    src_accum.reserve(num_approx);
    dst_accum.reserve(num_approx);
    if (enable_submap_matching)
    {
        for (int i = src_idx - submap_range; i < src_idx + submap_range + 1; ++i)
        {
            if (i >= 0 && i < static_cast<int>(keyframes.size() - 1))
            {
                src_accum += transformPcd(keyframes[i].pcd_, keyframes[i].pose_corrected_eig_);
            }
        }
        for (int i = dst_idx - submap_range; i < dst_idx + submap_range + 1; ++i)
        {
            if (i >= 0 && i < static_cast<int>(keyframes.size() - 1))
            {
                dst_accum += transformPcd(keyframes[i].pcd_, keyframes[i].pose_corrected_eig_);
            }
        }
    }
    else
    {
        src_accum = transformPcd(keyframes[src_idx].pcd_, keyframes[src_idx].pose_corrected_eig_);
        if (enable_quatro)
        {
            dst_accum = transformPcd(keyframes[dst_idx].pcd_, keyframes[dst_idx].pose_corrected_eig_);
        }
        else
        {
            // For ICP matching,
            // empirically scan-to-submap matching works better
            for (int i = dst_idx - submap_range; i < dst_idx + submap_range + 1; ++i)
            {
                if (i >= 0 && i < static_cast<int>(keyframes.size() - 1))
                {
                    dst_accum += transformPcd(keyframes[i].pcd_, keyframes[i].pose_corrected_eig_);
                }
            }
        }
    }
    return {*voxelizePcd(src_accum, voxel_res), *voxelizePcd(dst_accum, voxel_res)};
}

RegistrationOutput LoopClosure::icpAlignment(const pcl::PointCloud<PointType> &src,
                                             const pcl::PointCloud<PointType> &dst)
{
    RegistrationOutput reg_output;
    aligned_.clear();
    // merge subkeyframes before ICP
    pcl::PointCloud<PointType>::Ptr src_cloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr dst_cloud(new pcl::PointCloud<PointType>());
    *src_cloud = src;
    *dst_cloud = dst;
    nano_gicp_.setInputSource(src_cloud);
    nano_gicp_.calculateSourceCovariances();
    nano_gicp_.setInputTarget(dst_cloud);
    nano_gicp_.calculateTargetCovariances();
    nano_gicp_.align(aligned_);

    // handle results
    reg_output.score_ = nano_gicp_.getFitnessScore();
    // if matchness score is lower than threshold, (lower is better)
    if (nano_gicp_.hasConverged() && reg_output.score_ < config_.gicp_config_.icp_score_thr_)
    {
        reg_output.is_valid_ = true;
        reg_output.is_converged_ = true;
        reg_output.pose_between_eig_ = nano_gicp_.getFinalTransformation().cast<double>();
    }
    return reg_output;
}

/**
 * @brief 使用粗到精的方式对两个点云进行配准。
 *
 * 此函数分为两步：
 * 1. 使用 Quatro 算法进行粗配准，获得初步对齐的变换矩阵。
 * 2. 使用 ICP（迭代最近点）算法进行精配准，以优化对齐结果。
 *
 * @param src 源点云，需要进行对齐的点云。
 * @param dst 目标点云，作为对齐参考的点云。
 * @return RegistrationOutput 包含对齐结果的结构体，包括最终变换矩阵和是否收敛的状态。
 */
RegistrationOutput LoopClosure::coarseToFineAlignment(const pcl::PointCloud<PointType> &src,
                                                      const pcl::PointCloud<PointType> &dst)
{
    RegistrationOutput reg_output; // 用于存储配准结果的结构体。
    coarse_aligned_.clear();       // 清空用于存储粗配准结果的点云。

    // 第一步：使用 Quatro 算法进行粗配准。
    reg_output.pose_between_eig_ = (quatro_handler_->align(src, dst, reg_output.is_converged_));

    // 检查粗配准是否收敛。
    if (!reg_output.is_converged_)
    {
        // 如果配准失败，直接返回结果，标志未收敛。
        return reg_output;
    }
    else // 如果粗配准成功，
    {
        // 第二步：基于 Quatro 的结果进行精配准。
        // 将粗配准结果应用于源点云，生成粗配准后的点云。
        coarse_aligned_ = transformPcd(src, reg_output.pose_between_eig_);

        // 使用 ICP 算法对粗配准结果进行优化。
        const auto &fine_output = icpAlignment(coarse_aligned_, dst);

        // 保存 Quatro 算法的初步变换矩阵。
        const auto quatro_tf_ = reg_output.pose_between_eig_;

        // 更新配准结果为精配准的输出。
        reg_output = fine_output;

        // 合并粗配准（Quatro）和精配准（ICP）的变换矩阵，获得最终变换结果。
        reg_output.pose_between_eig_ = fine_output.pose_between_eig_ * quatro_tf_;
    }

    // 返回最终的配准结果。
    return reg_output;
}

/**
 * @brief 执行回环检测并进行点云配准
 *
 * 该函数通过将当前关键帧与历史关键帧进行比对，判断是否存在回环，并通过不同方法（Quatro + NANO-GICP 或 GICP）进行点云配准。
 *
 * @param query_keyframe 当前的查询关键帧，用于与历史关键帧进行比对。
 * @param keyframes 历史关键帧集合。
 * @param closest_keyframe_idx 与当前关键帧最近的历史关键帧索引。
 * @return RegistrationOutput 包含配准结果的结构体，包括是否收敛和最终变换矩阵。
 */
RegistrationOutput LoopClosure::performLoopClosure(const PosePcd &query_keyframe,
                                                   const std::vector<PosePcd> &keyframes,
                                                   const int closest_keyframe_idx)
{
    RegistrationOutput reg_output;                // 用于存储配准结果的结构体。
    closest_keyframe_idx_ = closest_keyframe_idx; // 保存最近的历史关键帧索引。

    // 检查最近的关键帧索引是否有效。
    if (closest_keyframe_idx_ >= 0)
    {
        // 调用函数生成源点云（src_cloud）和目标点云（dst_cloud）。
        const auto &[src_cloud, dst_cloud] = setSrcAndDstCloud(keyframes,
                                                               query_keyframe.idx_,              // 查询关键帧索引。
                                                               closest_keyframe_idx_,            // 最近历史关键帧索引。
                                                               config_.num_submap_keyframes_,    // 使用的子地图关键帧数量。
                                                               config_.voxel_res_,               // 体素分辨率。
                                                               config_.enable_quatro_,           // 是否启用 Quatro 粗配准。
                                                               config_.enable_submap_matching_); // 是否启用子地图匹配。

        // 将源点云和目标点云存储以便于可视化。
        *src_cloud_ = src_cloud;
        *dst_cloud_ = dst_cloud;

        // 判断是否启用 Quatro 算法进行粗到精的配准。
        if (config_.enable_quatro_)
        {
            // 输出调试信息，显示配准的点云规模。
            std::cout << "\033[1;35mExecute coarse-to-fine alignment: " << src_cloud.size()
                      << " vs " << dst_cloud.size() << "\033[0m\n";
            // 调用粗到精配准方法进行点云对齐。
            return coarseToFineAlignment(src_cloud, dst_cloud);
        }
        else
        {
            // 输出调试信息，显示配准的点云规模。
            std::cout << "\033[1;35mExecute GICP: " << src_cloud.size() << " vs "
                      << dst_cloud.size() << "\033[0m\n";
            // 调用 GICP 算法进行点云对齐。
            return icpAlignment(src_cloud, dst_cloud);
        }
    }
    else
    {
        // 如果没有有效的最近关键帧索引，返回默认无效的结果。
        return reg_output; // 默认返回的结果中 `is_valid` 为 false。
    }
}

pcl::PointCloud<PointType> LoopClosure::getSourceCloud()
{
    return *src_cloud_;
}

pcl::PointCloud<PointType> LoopClosure::getTargetCloud()
{
    return *dst_cloud_;
}

pcl::PointCloud<PointType> LoopClosure::getCoarseAlignedCloud()
{
    return coarse_aligned_;
}

// NOTE(hlim): To cover ICP-only mode, I just set `Final`, not `Fine`
pcl::PointCloud<PointType> LoopClosure::getFinalAlignedCloud()
{
    return aligned_;
}

int LoopClosure::getClosestKeyframeidx()
{
    return closest_keyframe_idx_;
}
