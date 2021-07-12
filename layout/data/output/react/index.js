import React from "react";
import ReactDOM from "react-dom";
import {Block1,Block2,Block3,Block4,Block11,Block16,Block13,Block15} from "./blocks;"
class Main extends React.Component{
	
	render(){
		return (
			<Block1/>
			<Block2/>
			<Block3/>
			<Block4/>
			<Block11/>
			<Block16/>
			<Block13/>
			<Block15/>
		)
	}
}
ReactDOM.render(<Main />, document.getElementById('root'))