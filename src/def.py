def csv_trans(path,file_name,mode='train'):
    csv_features = []
    csv_path=f'{path}/{file_name}/{file_name}.csv'
    df=pd.read_csv(csv_path)[csv_feature_dict.keys()]
    df = df.replace('-', 0)
    # MinMax scaling
    for col in df.columns:
        for col in df.columns:
            df[col] = df[col].astype(float) - csv_feature_dict[col][0]
            df[col] = df[col] / (csv_feature_dict[col][1] - csv_feature_dict[col][0])

    # zero padding
    pad = np.zeros((max_len, len(df.columns)))
    length = min(max_len, len(df))
    pad[:length] = df.to_numpy()[:length]

    # transpose to sequential data
    csv_feature = pad.T
    csv_features[i] = csv_feature

    if mode == 'train':
        json_path = f'{path}/{file_name}/{file_name}.json'
        with open(json_path, 'r') as f:
            json_file = json.load(f)

        crop = json_file['annotations']['crop']
        disease = json_file['annotations']['disease']
        risk = json_file['annotations']['risk']
        label = f'{crop}_{disease}_{risk}'

        return {'csv_feature': csv_feature),
                'label': label_encoder[label])


def image_trans(path,file_name,mode='train'):
    jpg_path=f'{path}/{file_name}/{file_name}.jpg'
    img = cv2.imread(image_path)
    img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255
    img = np.transpose(img, (2, 0, 1))

    if mode == 'train':
        json_path = f'{path}/{file_name}/{file_name}.json'
        with open(json_path, 'r') as f:
            json_file = json.load(f)

        crop = json_file['annotations']['crop']
        disease = json_file['annotations']['disease']
        risk = json_file['annotations']['risk']
        label = f'{crop}_{disease}_{risk}'

        return {
            'img': img,
            'label': label_encoder[label])