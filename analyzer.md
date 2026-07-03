# 因子信号分析工具

## 1. 角色设定
你现在是一个资深的 Python 自动化工程师。你的代码风格应该是：简洁、高健壮性、包含充分的注释，并且遵循 PEP 8 规范，变量及函数名应尽量使用完整的英文单词，单词之间使用下划线连接。

## 2. 任务背景与目标
我们现在有一个因子的信号数据,数据样例可以从"/dfs/data/tools/analyzer/example_signal"获取,我希望你编写一个工具以完成以下工作:
	1. 计算因子每个月在不同的universe上的hold,以及因子信号在不同universe上的统计数据,例如mean,std,skewness,kurtosis
	2. 计算因子在每个月于不同universe上的收益
	3. 可以同时计算多个因子在每个月于不同universe上的收益,并绘制收益曲线和对比多个因子在同一universe上的收益的差异

  
## 3. 工作目录与文件结构
- **工作目录：** `/dfs/data/tools/analyzer/`
- **因子信号数据文件：** `/dfs/data/tools/analyzer/example_signal/` （存放所有原始的 `.npz` 文件）
- **输出目录：** `/dfs/data/tools/analyzer/output/` （用于存放各种输出）
- **日志文件：** `/dfs/data/tools/analyzer/process.log` （记录处理进度和错误）
- **配置文件：** `/dfs/data/tools/analyzer/config.json` （用于配置分析参数）
- **关键数据: ** '/dfs/dataset/230-1724661625521/data/dataprod' (包含所有股票的分类信息等关键数据,基本都可以使用np.memmap进行读取)
- **关键数据说明: ** '192.168.0.152/wiki/doku.php?id=投研服务:数据服务:数据字典' (包含了关键数据中的每个字段的说明)


## 4. 注意事项
不同的因子的信号所覆盖的股票数可能不同,在每一个npz文件中都有一个名为"vStockCode"的np.array,这个array中包含了所有股票的代码,每个股票的代码用一个字符串表示,例如"000001.SH"。当前whole universe的股票代码可以从"/dfs/dataset/230-1724661625521/data/dataprod/__universe/uid.N128C"中获得,其dtype为U32,注意某些因子可能覆盖了whole universe中不存在的股票,当出现这种情况时将这些股票忽略处理。whole universe的所有交易日可以从"/dfs/dataset/230-1724661625521/data/dataprod/__universe/dates.N128C"中获取，注意不同的因子的信号所覆盖的时间可能存在不同,但是都会被包含在dates内。
所有universe的成分信息都可以在"/dfs/dataset/230-1724661625521/data/dataprod/“目录下获得,我希望你进行进行计算的universe有:
	1. whole universe
	2. A500
	3. ZZ100
	4. ZZ500
	5. ZZ800
	6. ZZ1000
	7. ZZ1500
	8. ZZ1800
	9. ZZ2000
	10. ZZ3000
	11. HS300
股票的未处理数据可以从"/dfs/dataset/230-1724661625521/data/dataprod/IntervalFull“文件夹中获得,复权后的数据可以从"/dfs/dataset/230-1724661625521/data/dataprod/ForwardAdjPrices"文件夹中获得,你可以使用这些数据来计算因子的收益和hold。
我希望有一个可视化的ui进行分析,用户可以在ui中选择不同的因子信号文件,以及不同的universe,然后查看因子在不同universe上的收益与hold,以及因子信号在不同universe上的统计数据,例如mean,std,skewness,kurtosis,同时最好能够对两个不同因子的信号在重叠的覆盖时间上进行对比。你拥有'/dfs/data/tools/analyzer'文件夹下所有文件的全部权限,拥有'/dfs/dataset/230-1724661625521/data/dataprod'文件夹下所有文件的读取权限但不允许对其中的任何文件进行编辑。如果'192.168.0.152'向你索要登录信息,你可以使用用户名public,密码wiki进行登录。

## 5. 执行要求
1. 在写代码之前，请先检查本地是否有需要的第三方库（如 `pandas`），如果没有，请向我申请安装权限。
2. 生成代码后，请询问我是否需要你直接在终端试运行一次。
