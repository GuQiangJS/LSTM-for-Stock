function setClass(ele,value)
{
	if(value>1)
	{
		ele.classList.add("td_up")
	}
	else if(value<1)
	{
		ele.classList.add("td_down")
	}
}
// Basic example
$(document).ready(function () {
$('#dtBasicExample').DataTable({
	"searching":false,
    "pageLength": 15,
	"lengthChange":false
});
//$('.dataTables_length').addClass('bs-select');
});
$(".price").each(function(){
	s=parseFloat(this.innerText);
    this.innerText=numeral(s).format('0.0000')
})
$(".percent").each(function(){
	setClass(this,parseFloat(this.innerText));
    this.innerText=numeral(this.innerText/100).format('0.00%')
})