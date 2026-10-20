# Host crash diagnostics — WSL2 box (collected 2026-10-20, read-only from WSL via powershell.exe)

**Headline:** two `0x133` DPC_WATCHDOG_VIOLATION BSODs (2026-10-19, 2026-10-20) both coincide with a
`stornvme` **controller error on `\Device\RaidPort2`**, which maps to the **WD_BLACK SN850X** (E: — the WSL
VHDX + Flash-Next model store). The SN850X controller throws Event 11 errors **near-daily** (mostly without a
crash); heavy Flash-Next NVMe mmap load escalates one into a DPC-watchdog bugcheck. Not ASPM (already Off on
AC), not a hardware machine-check (no WHEA), not the GPU (no TDR). The problem is the **SN850X NVMe
controller/driver/firmware (or its M.2 slot/thermals)**, not the model files or the qServer software.

## Drives
| Disk | Model | Firmware | Health | Bus | Role | NVMe controller | RaidPort |
|---|---|---|---|---|---|---|---|
| 0 | KINGSTON SKC3000D2048G (1908 GB) | EIFK31.6 | Healthy/OK | NVMe, **PCI bus 2** | **C: + D:** (system, /mnt/d) | Standard NVMe (VEN_2646) | SCSIPort **1** → RaidPort1 (no errors) |
| 1 | **WD_BLACK SN850X HS 2000GB** (1863 GB) | **EO001UCO** | Healthy/OK | NVMe, **PCI bus 6** | **E:** — `E:\WSL\Ubuntu-22.04\ext4.vhdx` (944 GB) = WSL VHD + `~/ai/flashnext/models` | Standard NVMe (VEN_15B7 = Sandisk/WD) | SCSIPort **2** → **RaidPort2 (ALL the controller errors)** |

- **Bus/slot:** Kingston = PCI bus 2, SN850X = PCI bus 6. On a Z790 / i9-13900K this usually means the Kingston
  is a **CPU-direct M.2** and the SN850X is a **chipset/PCH M.2** (behind the DMI link) — *confirm against the
  motherboard manual*; bus numbers alone aren't definitive. If the SN850X is the chipset slot, DMI contention
  under sustained reads is one more thing that can tip its controller into a timeout. Moving the model store to
  the CPU-direct slot (or the Kingston) would test this.
- **`HealthStatus=Healthy`** is SMART's own pass/fail; it does **not** rule out a firmware/controller stall.

## NVMe driver
Both controllers use the **Microsoft inbox `stornvme`** driver, version **10.0.26100.9549** (no vendor driver).
A standard `stornvme` with a drive that keeps throwing controller errors points at **drive firmware / the M.2
slot / thermals** rather than a swappable third-party driver, but an OS/stornvme cumulative update is still worth
being current on.

## OS
**Windows 11 25H2**, build **26200.9711** (the `ProductName` registry value reads "Windows 10 Pro" — a known
Win11 registry quirk; the 26200 build is Win11 25H2). Edition: Professional.

## Power plan & PCIe ASPM
- Active plan: **Ultimate Performance** (`eefd1130-f455-447d-87ef-334ddc509738`).
- **PCI Express → Link State Power Management:** **AC = Off (0)**, DC = Maximum power savings (2). The box runs on
  AC, so **ASPM is already Off** — PCIe link-power transitions are **not** the cause. (Setting DC→Off too is
  harmless tidiness but irrelevant on a desktop.)

## Drive temperature / wear / error counters
**NOT collected — `Get-StorageReliabilityCounter` returned access-denied (needs an elevated/admin PowerShell).**
SN850X temperature under sustained read is the single most useful missing number (thermal throttling/stall is a
leading `0x133` cause on gen4 drives). → see Rohit step 4 (HWiNFO).

## 7-day event timeline (System log: stornvme / Ntfs / volmgr / Kernel-Power 41 / BugCheck 1001)
All `stornvme` Event 11 = **"controller error on \Device\RaidPort2"** (= the **SN850X**). RaidPort1 (Kingston):
none.

| Time (2026) | Event | What the box was doing |
|---|---|---|
| 10-16 10:36:05–10:37:52 | **3× stornvme-11** + **41 reboot** (10:36:07) | unexpected reboot — but **BugcheckCode=0** (no dump) → a hang/hard-reset, not the DPC BSOD |
| 10-16 18:29:57 | stornvme-11 (no reboot) | — |
| 10-17 10:53:22 | stornvme-11 (no reboot) | — |
| 10-18 09:25:27 | stornvme-11 (no reboot) | — |
| 10-19 11:11:13 | stornvme-11 (no reboot) | — |
| **10-19 23:51:16** | **stornvme-11 → volmgr-162 "Dump file generation succeeded" → 41 → 1001 `0x133`** | **BSOD #1** (likely a prior Flash-Next benchmark session) |
| **10-20 17:31:00** | **stornvme-11 → volmgr-162 → 41 → 1001 `0x133`** | **BSOD #2** — during `arm5_standalone.sh` (unload → cold-fault ~29 GiB expert/PLE from the SN850X) |

Takeaway: the SN850X controller errors are **chronic and load-independent** (errors on 10-17/10-18/10-19 daytime
with no crash); the two **crashes** happened specifically when a heavy Flash-Next cold-fault burst was hitting the
drive. The workload is the **trigger/escalator**, not the root cause.

## 90-day Kernel-Power 41 split (BSOD vs sudden power-off)
Total unexpected reboots in 90 days: **4.**
- **WITH bugcheck (BSOD): 2** — both `BugcheckCode=307` (= `0x133`): **2026-10-19 23:51**, **2026-10-20 17:31**.
- **WITHOUT bugcheck (sudden power-off / hard reset): 2** — **2026-10-12 15:06**, **2026-10-16 10:36** (`BugcheckCode=0`,
  no crash dump — power trip, manual hard-reset, or a hang the watchdog didn't catch; a **different** failure mode).

So the two `0x133` BSODs are **new this week** and distinct from the two earlier dumpless power-offs.

---

## Steps for Rohit (admin) — in order. (I changed nothing; all of the below needs elevation.)

**1. Identify the faulting driver from the crash dump.**
Install **WinDbg** (Microsoft Store), run it **as Administrator**, then:
- `File → Open dump file` → `C:\Windows\Minidump\102026-10328-01.dmp` (newest; there is also a 10-19 dump).
- In the command box: `!analyze -v`
- Paste back the **`MODULE_NAME`**, **`IMAGE_NAME`**, **`FAILURE_BUCKET_ID`**, and the **stack** (the lines after
  "STACK_TEXT:"). That tells us whether the stuck DPC is in `stornvme`/`storport` (confirms the drive path),
  `nvlddmkm`, a network driver, or something else. (Symbols: `.symfix` then `.reload` if it prompts.)

**2. Back up E:, then update the SN850X firmware.**
- Current firmware (as Windows reports it): **`EO001UCO`**. This string doesn't match WD's public naming, so
  confirm the real rev in **Sandisk/WD Dashboard** or **CrystalDiskInfo** (Windows can mangle the NVMe firmware
  field). WD_BLACK SN850X 1/2 TB normally use the **620xxx** family (latest public 1 TB = **620361WD**, Nov 2023);
  there is no reliable 2 TB "latest" online — Dashboard is authoritative.
- **Back up E: first** (the WSL VHDX + models live there) — a firmware flash can, rarely, brick the drive.
- Flash only if Dashboard offers a newer rev; forum consensus is "don't update unless it fixes your symptom," and
  a controller-stall fix is exactly that symptom, so this one is worth doing.

**3. Set PCIe Link State Power Management to Off (DC too, for completeness — AC is already Off).**
```
powercfg /setacvalueindex SCHEME_CURRENT SUB_PCIEXPRESS ASPM 0
powercfg /setdcvalueindex SCHEME_CURRENT SUB_PCIEXPRESS ASPM 0
powercfg /setactive SCHEME_CURRENT
```
(AC is already 0; this just makes it explicit and covers DC. Low expected impact since ASPM wasn't the cause, but
it removes a variable.)

**4. Watch SN850X temperature during a sustained read.**
Install **HWiNFO64**, open **Sensors**, find the **WD_BLACK SN850X** temperature (and "Drive Temperature 2" /
controller temp if shown). Then generate sustained reads from the drive — e.g. in WSL:
`cat ~/ai/flashnext/models/UD-IQ4_XS/*.gguf > /dev/null` (reads ~93 GB; **do this only after step 2, and stop if
it crashes**). Watch for the SN850X crossing ~70–80 °C or a throttle flag right before any stall. If it spikes,
the HS heatsink/airflow or the M.2 slot is implicated.

**Also useful (optional):** confirm the SN850X's M.2 slot (CPU vs chipset) in the motherboard manual against "PCI
bus 6"; try reseating the SN850X; and consider temporarily moving the WSL VHDX / model store to the Kingston
(CPU-direct, zero logged errors) to see if the crashes stop — the strongest single test of "is it this drive."

## Couldn't collect (and why)
- **Drive temperature / wear / error counters** — `Get-StorageReliabilityCounter` needs an elevated PowerShell
  (access-denied from the WSL-invoked non-admin shell). Covered by Rohit step 4 (HWiNFO).
- **`!analyze -v` on the minidump** — no debugger installed, and `C:\Windows\Minidump` isn't readable without
  admin. Covered by Rohit step 1.
- **Definitive CPU-vs-chipset M.2 mapping** — needs the motherboard manual; only the PCI bus numbers (2 vs 6) are
  observable from Windows.
- **"Latest" SN850X 2 TB firmware** — not reliably available online; Sandisk/WD Dashboard is authoritative (step 2).
