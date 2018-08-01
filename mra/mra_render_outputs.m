to_plot = 0;
to_save_dicom = 1;
val_list = get_val_list;
files = {'ColArt', 'ColCap', 'ColDel', 'ColEVen', 'ColLVen', 'IMG'};
dirs.out = 'F:/BACKUPS/MRA/to_doctors_Jul16';
if ~exist(dirs.out, 'dir'), mkdir(dirs.out), end
for file_idx = 0 : 4
    run = 20 : 24;
    epoch = [99, 79, 79, 69, 99];
    dirs.f = sprintf('F:/BACKUPS/MRA/run%03d', run(file_idx+1));
    
    lab = readNPY(sprintf('%s/val_lab_Epoch%03d.npy', dirs.f, epoch(file_idx+1)));
    pred = readNPY(sprintf('%s/val_pred_Epoch%03d.npy', dirs.f, epoch(file_idx+1)));
    lab(lab < 0) = 0;
    pred(pred < 0) = 0;
    lab_ = uint8(lab);
    pred_ = uint8(pred);
    
    for i = 1 : size(lab, 1)
%     for i = 1
        fprintf('%02d -- %02d: ', file_idx, i)
        dirs.out_fig = sprintf('%s/%s/figures', dirs.out, val_list{i});
        if ~exist(dirs.out_fig, 'dir'), mkdir(dirs.out_fig), end

        dirs.out_dcm = sprintf('%s/%s/dicom/%s', dirs.out, val_list{i}, files{file_idx+1});
        if ~exist(dirs.out_dcm, 'dir'), mkdir(dirs.out_dcm), end
        
        mlab = squeeze(lab(i, 1, :, :));
        mpred = squeeze(pred(i, 1, :, :));
        for j = 2 : size(lab, 1)
            mlab = cat(2, mlab, squeeze(lab(i, j, :, :)));
            mpred = cat(2, mpred, squeeze(pred(i, j, :, :)));
        end
        L1_loss = sum(abs(mlab(:) - mpred(:))) / prod(size(mlab));
        disp(L1_loss)
        mout = cat(1, mlab, mpred);
        if to_plot
            fprintf('rendering output images... ')
            %             filename = sprintf('%s/pred%03d_%.2f', dirs.out_fig, i, L1_loss);
            filename = strrep(sprintf('%s/%s', dirs.out_fig, files{file_idx+1}), ' ', ' ');
            h = fig('units','inches','width',17,'height', 2.5,'font','Helvetica','fontsize',16, 'visibility', 'on');
            imshow(mout), colormap('jet'), axis('equal'), axis('off');
            title(sprintf('Top: Ground truth, Bottom: Prediction, L1 loss: %.2f', L1_loss))
            print(sprintf('%s.png', filename), '-dpng',  '-r700');
            close('all')
        end
        if to_save_dicom
            fprintf('saving dicom files... \n')
            filename = strrep(sprintf('%s/', dirs.out_dcm), ' ', ' ');
            tmp = permute(squeeze(pred_(:, :, i)), [2, 3, 1]);
            dicom_write_volume(tmp, sprintf('%s/IM.dcm', filename))
        end
    end
end
fprintf('\nDone.\n')

function val_list = get_val_list()
val_list = {'C:\\Workspace\\KU AIS Patient Anonymized//IAT AIS_pros\\BP34 20171106 1107 L dICA occ\\20171106 PRE-IAT\\AX_PWI_PA_0092_DSC_Collateral\\',
    'C:\\Workspace\\KU AIS Patient Anonymized//IAT AIS_pros\\BP37 20160920 R M1 occ\\20160920 PRE-IAT\\AX_PWI_PA5_DSC_Collateral\\',
    'C:\\Workspace\\KU AIS Patient Anonymized//IAT AIS_pros\\BP39 20170226 Lt M1 occ\\20170226 PRE-IAT\\AX_PWI_PA5_DSC_Collateral\\',
    'C:\\Workspace\\KU AIS Patient Anonymized//IAT AIS_pros\\BP41 20170824 0825 L dICA\\20170824 PRE-IAT\\AX_PWI_PA_0084_DSC_Collateral\\',
    'C:\\Workspace\\KU AIS Patient Anonymized//IAT AIS_pros\\BP44 20171004 1005 R dICA T occ\\20171004 PRE-IAT\\AX_PWI_PA_0084_DSC_Collateral\\',
    'C:\\Workspace\\KU AIS Patient Anonymized//IAT AIS_pros\\BP48 20171016 1020 L M2 MCA occ\\20171016 PRE-IAT\\AX_PWI_PA_DSC_Collateral\\',
    'C:\\Workspace\\KU AIS Patient Anonymized//IAT AIS_pros\\BP49 20170426 L M1 occ\\20170426 PRE-IAT\\AX_PWI5_DSC_Collateral\\',
    'C:\\Workspace\\KU AIS Patient Anonymized//IAT AIS_pros\\BP50 20171028 1030 L dICA occ\\20171028 PRE-IAT\\AX_PWI_PA_0082_DSC_Collateral\\',
    'C:\\Workspace\\KU AIS Patient Anonymized//IAT AIS_pros\\BP54 20170224 Rt pICA M1 occ\\20170224 PRE-IAT\\AX_PWI5_DSC_Collateral\\',
    'C:\\Workspace\\KU AIS Patient Anonymized//IAT AIS_pros\\BP60 20171018 1019 R MCA dissect\\20171018 PRE-IAT\\AX_PWI_PA_0084_DSC_Collateral\\',
    'C:\\Workspace\\KU AIS Patient Anonymized//IAT AIS_pros\\BP64 20170527 R M1 occ\\20170527 PRE-IAT\\AX_PWI5 DSC_Collateral_fail\\',
    'C:\\Workspace\\KU AIS Patient Anonymized//IAT AIS_pros\\BP67 20170823 0824 BA occ\\20170823 PRE-IAT\\AX_PWI_PA_0087_DSC_Collateral\\',
    'C:\\Workspace\\KU AIS Patient Anonymized//IAT AIS_pros\\BP69 20161001 L ICA occ\\20161001 PRE-IAT\\AX_PWI_PA5_DSC_Collateral\\',
    'C:\\Workspace\\KU AIS Patient Anonymized//IAT AIS_pros\\BP70 20170912 0913 R dICA occ\\20170912 PRE-IAT\\AX_PWI_PA_0087_DSC_Collateral\\',
    'C:\\Workspace\\KU AIS Patient Anonymized//IAT AIS_pros\\BP76 20170519 R dICA-MCA occ\\20170519 PRE-IAT\\AX_PWI5_DSC_Collateral\\',
    'C:\\Workspace\\KU AIS Patient Anonymized//IAT AIS_pros\\BP83 20170725 L M1 occ\\20170725 PRE-IAT\\AX_PWI_PA_0086_DSC_Collateral\\',
    'C:\\Workspace\\KU AIS Patient Anonymized//IAT AIS_pros\\BP94 20161123 R PCoA multiM2 occ\\20161123 PRE-IAT\\AX_PWI5_DSC_Collateral\\'};
val_list = strrep(val_list, '\\', '/');
val_list = strrep(val_list, '//', '/');
val_list = strrep(val_list, 'C:/Workspace/KU AIS Patient Anonymized/', '');
end