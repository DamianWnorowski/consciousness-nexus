param([Parameter(ValueFromRemainingArguments=$true)]$args)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$scriptDir\consciousness-cli" @args
