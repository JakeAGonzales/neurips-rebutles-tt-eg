# !/bin/bash

deepspeed --master_port=12345 online_ipo_1.py --lora
deepspeed --master_port=12345 online_ipo_2.py --lora
deepspeed --master_port=12345 extragradient.py --lora

# deepspeed --master_port=12345 nash_md.py --lora
# deepspeed --master_port=12345 nash_md_pg.py --lora
