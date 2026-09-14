"""Give this instance a new public IP, for when YouTube refuses the current one.

Measured 2026-09-13: a datacenter IP served about 100 track downloads and was then refused with
HTTP 403 on every player request. It happened at 10 downloads per minute and again at 3, so the
limit is a count per address, not a rate, and pacing does not avoid it. Waiting did not clear it
either (85 minutes, five probes, no lift). A different address does.

Allocate a new Elastic IP, associate it with this instance, which replaces the current public
address, then release the previous one. About ten seconds, with no reboot: the run keeps its state,
tmux survives, and the gate's next probe goes out from the new address. Your ssh session to the old
address drops; reconnect on the new one.

Needs an instance role allowing ec2:DescribeAddresses, AllocateAddress, AssociateAddress and
ReleaseAddress. An allocated Elastic IP is billed whether or not it is attached, so the old one is
always released and a half-finished rotation releases the one it just took.
"""

import logging
import time
import urllib.request

log = logging.getLogger("djdata.rotate")
IMDS = "http://169.254.169.254/latest"


def _imds(path: str, timeout: float = 2.0) -> str:
    """Read one instance metadata value. IMDSv2: take a token, then use it."""
    token = urllib.request.urlopen(
        urllib.request.Request(f"{IMDS}/api/token", method="PUT",
                               headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}), timeout=timeout).read()
    req = urllib.request.Request(f"{IMDS}/meta-data/{path}", headers={"X-aws-ec2-metadata-token": token})
    return urllib.request.urlopen(req, timeout=timeout).read().decode()


def public_ip() -> str:
    return _imds("public-ipv4")


def rotate() -> str:
    """Swap this instance's public address. Returns the new IP. Raises if it could not be done."""
    import boto3

    instance_id = _imds("instance-id")
    region = _imds("placement/region")
    ec2 = boto3.client("ec2", region_name=region)
    before = public_ip()

    held = ec2.describe_addresses(Filters=[{"Name": "instance-id", "Values": [instance_id]}])["Addresses"]
    old_allocation = held[0]["AllocationId"] if held else None      # None on the first rotation: the IP is AWS-assigned

    new = ec2.allocate_address(Domain="vpc")
    try:
        ec2.associate_address(AllocationId=new["AllocationId"], InstanceId=instance_id, AllowReassociation=True)
    except Exception:
        ec2.release_address(AllocationId=new["AllocationId"])
        raise
    if old_allocation:
        try:
            ec2.release_address(AllocationId=old_allocation)
        except Exception as e:
            # the new address is already live, so the rotation worked; a stranded address is billed
            # at a few cents a day and counts against the region's quota, so it is logged loudly.
            log.error("could not release the old Elastic IP %s: %s -- release it by hand", old_allocation, e)

    for _ in range(30):
        time.sleep(1.0)
        try:
            if public_ip() == new["PublicIp"]:
                break
        except Exception:
            pass        # metadata is briefly unreachable while the address changes
    log.warning("public IP rotated: %s -> %s", before, new["PublicIp"])
    return new["PublicIp"]
