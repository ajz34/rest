

# pyrest

`pyrest` 是一个用于量子化学计算的开源软件项目，提供了多种基组、有效芯势（ECP）以及DFT网格生成等功能。该项目支持多种计算类型，包括Hartree-Fock（HF）、Kohn-Sham DFT、MP2、RPA、FCIQMC等，并且支持并行计算以提高性能。

## 项目结构

- **`basis-set-pool`**：包含各种基组数据，如 `6-31+G*`, `cc-pV5Z`, `def2-SVP` �{...}等，适用于不同元素。
- **`src`**：项目的主要源代码目录，包含多个模块：
  - `basis_io`：处理基组数据的输入输出。
  - `check_norm`：处理态的占据规范化。
  - `constants`：量子化学计算中使用的常量和数学表。
  - `ctrl_io`：控制输入输出，处理输入参数和几何信息。
  - `dft`：DFT相关功能，包括网格生成和XC泛函。
  - `geom_io`：处理几何结构的输入输出。
  - `scf`：自洽场（SCF）计算。
  - `tensor`：张量计算模块。
  - `util`：通用工具模块。

## 主要功能

- 支持多种基组和ECP，适用于广泛的元素。
- DFT网格生成，包括径向和角向网格。
- 支持并行计算以加速张量运算。
- 提供了多种量子化学计算方法，如HF、DFT、MP2、RPA等。
- 支持从JSON或TOML文件中解析输入参数。

## 使用示例

```rust
// 读取基组信息
let basis = Basis4Elem::parse_json_from_file("path/to/basis.json", &CintType::Cint);

// 初始化输入参数
let mut ctrl = InputKeywords::init_ctrl();

// 设置打印级别和线程数
ctrl.py_set_print_level(2);
ctrl.py_set_num_threads(4);

// 生成原子网格
let (grids, weights) = atom_grid("path/to/basis", 1e-6, 50, 100, vec![1], 0, (0.0, 0.0, 0.0), 3, "default".to_string(), "treutler".to_string(), 2);

// 执行SCF计算
let scf_result = scf::run_scf(&basis, &ctrl, &grids, &weights);
```

## 许可证

该项目遵循 MIT 许可证。详情请查看 [LICENSE](LICENSE) 文件。

## 构建与测试

- 使用 `Cargo` 构建项目：
  ```bash
  cargo build --release
  ```
- 使用 `cargo test` 执行测试：
  ```bash
  cargo test
  ```

## 文档

更多详细文档可在项目页面找到，包括详细的API参考和计算方法说明。

## 贡献

我们欢迎任何形式的贡献。如果您有兴趣参与，请先阅读我们的 [贡献指南](CONTRIBUTING.md)。