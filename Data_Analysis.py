# 读取数据：
    # 数据路径
    data_path = RAW_DATA_PATH / 'news_recommendation/'
    # 训练集
    # pd.read_csv用于从文件中读取数据并将其加载到一个pandas.DataFrame对象中，DataFrame是pandas库中用于存储和操作结构化数据的主要数据结构。
    trn_click = pd.read_csv(data_path / 'train_click_log.csv') #用于训练的交互记录
    item_df = pd.read_csv(data_path / 'articles.csv') # 物品信息
    item_df = item_df.rename(columns={'article_id': 'click_article_id'})  #重命名，方便后续match
    item_emb_df = pd.read_csv(data_path / 'articles_emb.csv') # 物品embedding
    # 测试集
    tst_click = pd.read_csv(data_path / 'testA_click_log.csv') # 用于测试的交互记录

# 数据预处理：
    # 对每个用户的点击时间戳进行排序
    trn_click['rank'] = trn_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)
    tst_click['rank'] = tst_click.groupby(['user_id'])['click_timestamp'].rank(ascending=False).astype(int)
    # 计算用户点击文章的次数，并添加新的一列count
    trn_click['click_cnts'] = trn_click.groupby(['user_id'])['click_timestamp'].transform('count')
    # transform('count')：计算每个用户点击文章的总次数，并将结果广播到每一行。
    tst_click['click_cnts'] = tst_click.groupby(['user_id'])['click_timestamp'].transform('count')
    # train_click_log.csv文件数据中每个字段的含义:
    # user_id: 用户的唯一标识
    # click_article_id: 用户点击的文章唯一标识
    # click_timestamp: 用户点击文章时的时间戳
    # click_environment: 用户点击文章的环境
    # click_deviceGroup: 用户点击文章的设备组
    # click_os: 用户点击文章时的操作系统
    # click_country: 用户点击文章时的所在的国家
    # click_region: 用户点击文章时所在的区域
    # click_referrer_type: 用户点击文章时，文章的来源

    # 训练集：
    trn_click.describe()  # 返回每列属性的统计型数据
    trn_click.user_id.nunique() # 返回200000    训练集中的用户数量为20w
    trn_click.groupby('user_id')['click_article_id'].count().min()  # 返回2  训练集里面每个用户至少点击了两篇文章
    # 画直方图大体看一下基本的属性分布
    plt.figure()
    plt.figure(figsize=(15, 20))
    i = 1
    for col in ['click_article_id', 'click_timestamp', 'click_environment', 'click_deviceGroup', 'click_os', 'click_country',
                'click_region', 'click_referrer_type', 'rank', 'click_cnts']:
        plot_envs = plt.subplot(5, 2, i)
        i += 1
        v = trn_click[col].value_counts().reset_index()[:10]
        # Use iloc to access columns by position to avoid column name issues
        fig = sns.barplot(x=v.iloc[:, 0], y=v.iloc[:, 1])
        for item in fig.get_xticklabels():
            item.set_rotation(90)
        plt.title(col)
    plt.tight_layout()
    plt.show()
    # 从点击时间clik_timestamp来看，分布较为平均，可不做特殊处理。由于时间戳是13位的，后续将时间格式转换成10位方便计算。
    # 从点击环境click_environment来看，仅有1922次（占0.1%）点击环境为1；仅有24617次（占2.3%）点击环境为2；剩余（占97.6%）点击环境为4。
    # 从点击设备组click_deviceGroup来看，设备1占大部分（60.4%），设备3占36%。

    # 测试集：
    tst_click.describe() # 我们可以看出训练集和测试集的用户是完全不一样的, 训练集的用户ID由0-199999，而测试集A的用户ID由200000-249999。
    tst_click.user_id.nunique() #测试集中的用户数量为5w
    tst_click.groupby('user_id')['click_article_id'].count().min() # 注意测试集里面有只点击过一次文章的用户

    # 物品信息：
    item_df.head()
    item_df['words_count'].value_counts()
    print(item_df['category_id'].nunique())     # 461个文章主题
    item_df['category_id'].hist()
    item_df.shape       # 364047篇文章
    item_emb_df.shape # (364047, 251)

# 综合分析
    user_click_merge = pd.concat([trn_click, tst_click])
    #用户重复点击
    user_click_count = user_click_merge.groupby(['user_id', 'click_article_id'])['click_timestamp'].agg({'count'}).reset_index()
    user_click_count[:5] # 查看前五行
    user_click_count[user_click_count['count']>7]
    user_click_count['count'].unique() # array([ 1,  2,  4,  3,  6,  5, 10,  7, 13])

    #用户点击新闻次数
    user_click_count.loc[:,'count'].value_counts() # 可以看出：有1605541（约占99.2%）的用户未重复阅读过文章，仅有极少数用户重复点击过某篇文章。 这个也可以单独制作成特征
    # count
    # 1     1605541
    # 2       11621
    # 3         422
    # 4          77
    # 5          26
    # 6          12
    # 10          4
    # 7           3
    # 13          1
    # Name: count, dtype: int64

    # 用户点击环境变化分析
    def plot_envs(df, cols, r, c):
        plt.figure()
        plt.figure(figsize=(10, 5))
        i = 1
        for col in cols:
            plt.subplot(r, c, i)
            i += 1
            v = df[col].value_counts().reset_index()
            fig = sns.barplot(x=v.iloc[:, 0], y=v.iloc[:, 1])
            for item in fig.get_xticklabels():
                item.set_rotation(90)
            plt.title(col)
        plt.tight_layout()
        plt.show()
    # 分析用户点击环境变化是否明显，这里随机采样5个用户分析这些用户的点击环境分布
    sample_user_ids = np.random.choice(tst_click['user_id'].unique(), size=5, replace=False)
    sample_users = user_click_merge[user_click_merge['user_id'].isin(sample_user_ids)]
    cols = ['click_environment','click_deviceGroup', 'click_os', 'click_country', 'click_region','click_referrer_type']
    for _, user_df in sample_users.groupby('user_id'):
        plot_envs(user_df, cols, 2, 3)  # 按ID分组并统计属性各类别次数
    # 可以看出绝大多数数的用户的点击环境是比较固定的。思路：可以基于这些环境的统计特征来代表该用户本身的属性

    # 用户点击新闻数量的分布
    user_click_item_count = sorted(user_click_merge.groupby('user_id')['click_article_id'].count().values, reverse=True)
    plt.plot(user_click_item_count)
    # 可以根据用户的点击文章次数看出用户的活跃度
    #点击次数在前50的用户
    plt.plot(user_click_item_count[:50])
    # 点击次数排前50的用户的点击次数都在100次以上。
    # 思路：我们可以定义点击次数大于等于100次的用户为活跃用户，这是一种简单的处理思路， 判断用户活跃度，更加全面的是再结合上点击时间，后面我们会基于点击次数和点击时间两个方面来判断用户活跃度。

    # 新闻点击次数分析
    item_click_count = sorted(user_click_merge.groupby('click_article_id')['user_id'].count(), reverse=True)
    plt.plot(item_click_count)
    plt.plot(item_click_count[:100])
    # 可以看出点击次数最多的前100篇新闻，点击次数大于1000次
    plt.plot(item_click_count[:20])
    # 可以看出点击次数最多的前20篇新闻，点击次数大于2500。
    # 思路：可以定义这些新闻为热门新闻， 这个也是简单的处理方式，后面我们也是根据点击次数和时间进行文章热度的一个划分。
    plt.plot(item_click_count[3500:])
    # 可以发现很多新闻只被点击过一两次。思路：可以定义这些新闻是冷门新闻。
    tmp = user_click_merge.sort_values('click_timestamp')
    tmp['next_item'] = tmp.groupby(['user_id'])['click_article_id'].transform(lambda x:x.shift(-1))
    union_item = tmp.groupby(['click_article_id','next_item'])['click_timestamp'].agg({'count'}).reset_index().sort_values('count', ascending=False)
    # 对每个用户组，使用 transform 方法和 lambda 函数，将 click_article_id 列的值向下移动一行（shift(-1)），从而得到每个用户的下一次点击文章ID。
    union_item[['count']].describe()
    # 由统计数据可以看出，平均共现次数2.88，最高为1687，说明用户看的新闻，相关性是比较强的。

    # 新闻文章信息
    # 不同类型的新闻出现的次数
    plt.plot(user_click_merge['category_id'].value_counts().values)
    #出现次数比较少的新闻类型, 有些新闻类型，基本上就出现过几次
    plt.plot(user_click_merge['category_id'].value_counts().values[150:])
    # 新闻字数
    plt.plot(user_click_merge['words_count'].values)

    # 用户点击的新闻类型的偏好 此特征可以用于度量用户的兴趣是否广泛。
    plt.plot(sorted(user_click_merge.groupby('user_id')['category_id'].nunique(), reverse=True)) # 从上图中可以看出有一小部分用户阅读类型是极其广泛的，大部分人都处在20个新闻类型以下。

    # 用户查看文章的长度的分布 通过统计不同用户点击新闻的平均字数，这个可以反映用户是对长文更感兴趣还是对短文更感兴趣。
    plt.plot(sorted(user_click_merge.groupby('user_id')['words_count'].mean(), reverse=True))
    # 从上图中可以发现有一小部分人看的文章平均词数非常高，也有一小部分人看的平均文章次数非常低。 大多数人偏好于阅读字数在200-400字之间的新闻。
    #挑出大多数人的区间仔细看看
    plt.plot(sorted(user_click_merge.groupby('user_id')['words_count'].mean(), reverse=True)[1000:45000]) # 可以发现大多数人都是看250字以下的文章

    # 用户点击新闻的时间分析
    #为了更好的可视化，这里把时间进行归一化操作
    from sklearn.preprocessing import MinMaxScaler
    mm = MinMaxScaler()
    user_click_merge['click_timestamp'] = mm.fit_transform(user_click_merge[['click_timestamp']])
    user_click_merge['created_at_ts'] = mm.fit_transform(user_click_merge[['created_at_ts']])
    # fit_transform 方法首先计算数据的最小值和最大值，然后将每个值归一化到 [0, 1] 区间。
    # 注意：fit_transform 方法需要输入一个二维数组，因此使用 [['click_timestamp']] 而不是 ['click_timestamp']。
    user_click_merge = user_click_merge.sort_values('click_timestamp')
    def mean_diff_time_func(df, col):
        df = pd.DataFrame(df, columns=[col])
        df['time_shift1'] = df[col].shift(1).fillna(0)
        df['diff_time'] = abs(df[col] - df['time_shift1'])
        return df['diff_time'].mean()
    # 这段代码定义了一个函数 mean_diff_time_func，用于计算 DataFrame 中某一列（通常表示时间戳）的相邻值之间的时间差的平均值
    # 点击时间差的平均值
    mean_diff_click_time = user_click_merge.groupby('user_id')[['click_timestamp', 'created_at_ts']].apply(lambda x: mean_diff_time_func(x, 'click_timestamp'))
    # 从上图可以发现不同用户点击文章的时间差是有差异的
    # 前后点击文章的创建时间差的平均值
    mean_diff_created_time = user_click_merge.groupby('user_id')[['click_timestamp', 'created_at_ts']].apply(lambda x: mean_diff_time_func(x, 'created_at_ts'))
    # 从图中可以发现用户先后点击文章，文章的创建时间也是有差异的

    # 用户前后点击文章的相似性分布
    item_idx_2_rawid_dict = dict(zip(item_emb_df['article_id'], item_emb_df.index))
    del item_emb_df['article_id']
    item_emb_np = np.ascontiguousarray(item_emb_df.values, dtype=np.float32)
    # 随机选择5个用户，查看这些用户前后查看文章的相似性
    sub_user_ids = np.random.choice(user_click_merge.user_id.unique(), size=15, replace=False)
    sub_user_info = user_click_merge[user_click_merge['user_id'].isin(sub_user_ids)]
    sub_user_info.head()
    def get_item_sim_list(df):
        sim_list = []
        item_list = df['click_article_id'].values
        for i in range(0, len(item_list)-1):
            emb1 = item_emb_np[item_idx_2_rawid_dict[item_list[i]]]
            emb2 = item_emb_np[item_idx_2_rawid_dict[item_list[i+1]]]
            sim_list.append(np.dot(emb1,emb2)/(np.linalg.norm(emb1)*(np.linalg.norm(emb2))))  # 用item embeding 计算余弦相似度
        sim_list.append(0)
        return sim_list
    for _, user_df in sub_user_info.groupby('user_id'):
        item_sim_list = get_item_sim_list(user_df)
        plt.plot(item_sim_list)
    # 从图中可以看出有些用户前后看的商品的相似度波动比较大，有些波动比较小，也是有一定的区分度的。

# 总结
    # 通过数据分析的过程， 我们目前可以得到以下几点重要的信息， 这个对于我们进行后面的特征制作和分析非常有帮助：
    # 训练集和测试集的用户id没有重复，也就是测试集里面的用户没有模型是见过的
    # 训练集中用户最少的点击文章数是2， 而测试集里面用户最少的点击文章数是1
    # 用户对于文章存在重复点击的情况， 但这个都存在于训练集里面
    # 同一用户的点击环境存在不唯一的情况，后面做这部分特征的时候可以采用统计特征
    # 用户点击文章的次数有很大的区分度，后面可以根据这个制作衡量用户活跃度的特征
    # 文章被用户点击的次数也有很大的区分度，后面可以根据这个制作衡量文章热度的特征
    # 用户看的新闻，相关性是比较强的，所以往往我们判断用户是否对某篇文章感兴趣的时候， 在很大程度上会和他历史点击过的文章有关
    # 用户点击的文章字数有比较大的区别， 这个可以反映用户对于文章字数的区别
    # 用户点击过的文章主题也有很大的区别， 这个可以反映用户的主题偏好
    # 不同用户点击文章的时间差也会有所区别， 这个可以反映用户对于文章时效性的偏好
    # 所以根据上面的一些分析，可以更好的帮助我们后面做好特征工程， 充分挖掘数据的隐含信息。




    # ### 每项分析的出发点、结果和对后续特征工程的帮助
    #
    # 1. **用户分布分析**
    #    - **出发点**：了解训练集和测试集的用户分布情况，确保模型在测试集上的泛化能力。
    #    - **结果**：训练集和测试集的用户ID没有重叠，测试集中的用户是模型未见过的。
    #    - **帮助**：这表明模型需要具备处理新用户的能力，不能依赖于用户的历史行为。
    #
    # 2. **用户重复点击分析**
    #    - **出发点**：了解用户是否对某些文章有特别的兴趣，表现为重复点击。
    #    - **结果**：训练集中存在用户重复点击文章的情况，但这种情况较少。
    #    - **帮助**：可以将用户是否重复点击某篇文章作为一个特征，表示用户对该文章的兴趣程度。
    #
    # 3. **用户点击环境分析**
    #    - **出发点**：了解用户在不同环境下的点击行为，判断用户是否在特定环境下更活跃。
    #    - **结果**：大多数用户的点击环境是固定的，但也有少数用户在不同的环境中点击文章。
    #    - **帮助**：可以将用户的点击环境作为一个特征，表示用户的行为偏好。
    #
    # 4. **用户点击文章次数分析**
    #    - **出发点**：了解用户的活跃度，判断用户是否是活跃用户。
    #    - **结果**：用户点击文章的次数分布差异较大，可以用来衡量用户的活跃度。
    #    - **帮助**：可以将用户点击文章的次数作为一个特征，表示用户的活跃度。
    #
    # 5. **文章被点击次数分析**
    #    - **出发点**：了解文章的热度，判断文章是否是热门文章。
    #    - **结果**：文章被点击的次数分布差异较大，可以用来衡量文章的热度。
    #    - **帮助**：可以将文章被点击的次数作为一个特征，表示文章的热度。
    #
    # 6. **用户点击文章时间差分析**
    #    - **出发点**：了解用户对文章时效性的偏好，判断用户是否更倾向于阅读最新的文章。
    #    - **结果**：用户点击文章的时间差分布差异较大，可以用来衡量用户对文章时效性的偏好。
    #    - **帮助**：可以将用户点击文章的时间差作为一个特征，表示用户对文章时效性的偏好。
    #
    # 7. **用户点击文章相似性分析**
    #    - **出发点**：了解用户兴趣的稳定性，判断用户是否对相似的文章更感兴趣。
    #    - **结果**：用户前后点击文章的相似性分布差异较大，可以用来衡量用户兴趣的稳定性。
    #    - **帮助**：可以将用户点击文章的相似性作为一个特征，表示用户兴趣的稳定性。
    #
    # ### 总结
    # 通过这些分析，我们可以得到以下重要信息，这些信息对于后续的特征工程和模型训练非常有帮助：
    # - **用户分布**：训练集和测试集的用户ID没有重叠，测试集中的用户是模型未见过的。
    # - **用户活跃度**：用户点击文章的次数分布差异较大，可以用来衡量用户的活跃度。
    # - **文章热度**：文章被点击的次数分布差异较大，可以用来衡量文章的热度。
    # - **用户兴趣稳定性**：用户前后点击文章的相似性分布差异较大，可以用来衡量用户兴趣的稳定性。
    # - **用户行为偏好**：用户的点击环境、文章字数偏好、文章主题偏好等可以作为特征，表示用户的行为偏好。
    #
    # 这些分析结果可以帮助我们更好地理解数据，为后续的特征工程提供依据，从而提高模型的性能和泛化能力。


# 数据分析（预处理）的作用：
        # 定特征中的超参数  比如定义活跃用户的阈值等
        # 了解数据分布 确定后续召回和排序的策略  例如有没有冷启动问题 等