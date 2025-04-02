function [joint_groups] = mpii_get_pck_auc_joint_groups()

joint_groups = { 'Head', [1,17];
                 'Neck', [2];
                 'Shou', [3,6];
                 'Elbow', [4,7];
                 'Wrist', [5,8];
                 'spine', [16];
                 'Hip', [9,12,15];
                 'Knee', [10,13];
                 'Ankle', [11,14];
                 };
end