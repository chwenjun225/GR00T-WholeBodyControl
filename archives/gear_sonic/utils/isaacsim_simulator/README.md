# Cầu nối Isaac Sim ↔ SONIC

Bản triển khai trên nền `bae92fd8135a06962d6672ab1e389d8731e3c2b7`.
Dành cho Isaac Sim 6.0.1, PhysX và một G1 29 khớp với hai bàn tay Dex3
cùng thuộc một hệ khớp. Các lời gọi được đối chiếu với mã NVIDIA v6.0.1,
nhưng chưa chạy thực tế trong Isaac Sim.

## Phạm vi

- Đọc cảnh USD có sẵn; không nhập lại URDF hoặc lưu đè cảnh.
- Kiểm tra cảnh dùng mét, trục Z hướng lên, có một PhysicsScene.
- Ánh xạ khớp bằng tên theo thứ tự phần cứng SONIC; thiếu khớp thì báo lỗi.
- Phát `rt/lowstate`, `rt/secondary_imu`, `rt/odostate` và trạng thái Dex3.
- Nhận `rt/lowcmd` và lệnh Dex3, tính mô-men PD cộng truyền thẳng.
- Giới hạn mô-men theo giá trị nhỏ hơn giữa cấu hình G1 và giới hạn USD.
- Khóa dữ liệu được tạo trước khi đăng ký hàm nhận DDS; lệnh được sao chép
  dưới khóa rồi mới chuyển tới vòng vật lý.
- Kiểm tra CRC của lệnh thân; chỉ hỗ trợ chế độ PR (pitch/roll), không hỗ trợ AB.
- Chờ lệnh thân đầu tiên trước khi tiến vật lý; sau đó mất lệnh quá 0,5 giây
  sẽ dừng và đóng mô phỏng. Mất lệnh tay đưa mô-men tay về không.
- Tần số vật lý mặc định 500 Hz; dựng hình mỗi 10 bước. Không bảo đảm máy
  đạt thời gian thực: nếu chậm trên 100 ms sẽ có thông báo.
- IMU lý tưởng: hướng quay wxyz, vận tốc góc trong hệ thân; gia tốc riêng
  ước lượng bằng sai phân vận tốc khối tâm rồi trừ trọng lực. Chưa mô phỏng
  nhiễu, độ lệch hoặc vị trí lắp cảm biến thật.
- `tau_est` là mô-men bộ truyền động đã đặt ở bước trước, không phải lực
  phản lực hay lực tiếp xúc đo được.

Các tệp `image_publish_utils.py`, `sensor_server.py`, `metric_utils.py`,
`sim_utils.py` và `robot.py` còn từ bản mẫu; đường chạy mới không sử dụng
chúng. Chưa tích hợp ảnh, tay điều khiển, dây đỡ đàn hồi, đánh giá tác vụ
hoặc tự động khởi động lại khi người dùng dừng dòng thời gian. Điểm vào
MuJoCo cũ `run_sim_loop.py` được giữ nguyên.

## Môi trường

Chạy từ gốc GR00T-WholeBodyControl bằng Python đóng gói cùng Isaac Sim.
Cần NumPy, PyYAML và Unitree SDK2 Python nhập được trong chính môi trường
đó. Không cần cài toàn bộ gói phụ thuộc huấn luyện, Pinocchio hoặc MuJoCo.

```bash
cd ~/projects/cibo/3rd_party/GR00T-WholeBodyControl
~/isaacsim/python.sh -c 'import numpy, yaml; from unitree_sdk2py.core.channel import ChannelFactoryInitialize'
```

Bộ điều khiển C++ hiện dùng miền DDS 0. Mặc định cầu nối cũng là miền 0,
giao diện nội bộ `lo`. Đổi miền của một phía không tự đổi phía còn lại.
Chỉ chạy một nguồn phát trạng thái G1 trên các kênh này.

## Bước 1: chỉ kiểm tra cảnh

Thay `/World/Robots/g1` bằng đường dẫn thật trong cảnh. Nếu không tồn tại,
thông báo lỗi liệt kê các gốc hệ khớp đã tìm thấy.

```bash
~/isaacsim/python.sh -m gear_sonic.scripts.run_isaacsim_loop \
  --usd-path ~/projects/cibo/contents/scenes/cibo.usd \
  --robot-path /World/Robots/g1 \
  --inspect
```

Kết quả cần có: bảng ánh xạ 29 khớp thân, 7 khớp mỗi tay, góc khớp,
tư thế gốc và thân trên. Chế độ này khởi tạo PhysX để lấy cấu trúc,
không mở DDS, không chạy vòng lặp điều khiển. Ứng dụng đóng khi in xong.

Nếu tên liên kết không phải `pelvis` và `torso_link`, truyền
`--base-path /duong/dan/pelvis` và `--torso-path /duong/dan/torso_link`.
Các đường dẫn phải là vật rắn thuộc cây robot.
Nếu cảnh chỉ có thân G1, dùng `--without-hands`.
Nếu hai bàn tay là các hệ khớp độc lập, bản này chưa hỗ trợ:
không bỏ qua lỗi thiếu khớp để tiếp tục điều khiển.

## Bước 2: phát trạng thái và chờ SONIC

Chạy lại lệnh trên nhưng bỏ `--inspect`. Cầu nối phát trạng thái trong khi
giữ nguyên thời gian vật lý và chờ lệnh thân đầu tiên. Khi đang chờ,
`tick` không tăng vì thời gian mô phỏng chưa tiến.

Ở cửa sổ lệnh khác, có thể chạy bộ kiểm tra sẵn có trong CIBO:

```bash
cd ~/projects/cibo
~/isaacsim/python.sh scripts/test_dds_loop.py \
  --domain 0 --interface lo --topic lowstate --duration 10
```

Bộ kiểm tra đó chưa kiểm tra IMU thân trên và bàn tay. Có bản tin không
đồng nghĩa mô phỏng đã điều khiển chính xác.

## Bước 3: nối SONIC

Sau khi kiểm tra ánh xạ và trạng thái, chạy trong cửa sổ lệnh khác:

```bash
cd ~/projects/cibo
bash scripts/run_sonic_controller.sh --model sonic_v1_1 --input-type keyboard
```

Lệnh này dùng tập lệnh đã có trong CIBO; không phải tệp được thêm bởi
thay đổi này. Nó có thể biên dịch và thiết lập phụ thuộc.
Khi SONIC phát lệnh thân hợp lệ, vật lý bắt đầu chạy. Chưa chứng minh
robot sẽ đứng ổn định trong cảnh của bạn: cần kiểm tra tư thế đầu, tiếp xúc,
khối lượng, quán tính và tham số bộ truyền động của USD.

Nếu tạm dừng/dừng dòng thời gian, đóng và khởi động lại cầu nối.
Đây là chương trình chạy độc lập; không dán phần khởi tạo SimulationApp
vào trình soạn thảo mã của một phiên Isaac Sim đã mở.

## Kiểm tra độc lập

```bash
python -m unittest discover -s gear_sonic/utils/isaacsim_simulator/tests -v
```

Kiểm tra thứ tự khớp với mã G1 gốc, ánh xạ hoán vị, mô-men PD và giới hạn,
quy ước IMU, hàm nhận ngay trong lúc khởi tạo, ảnh chụp lệnh độc lập,
đóng gói trạng thái/CRC, chờ lệnh đầu tiên và dừng khi lệnh cũ.
DDS và PhysX được thay bằng đối tượng giả trong các phép thử liên quan;
các phép thử không chứng minh kết nối mạng hay động lực học thực tế.

